from collections.abc import Sequence
import hashlib
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def probe_solver_features(
        self,
        obs: dict,
        *,
        noise_seed: int,
        trace_budgets: tuple[int, ...] = (),
    ) -> dict[str, np.ndarray]:
        """Return no-motion early-flow features for the PyTorch policy path."""
        if not self._is_pytorch_model:
            raise NotImplementedError("solver probing currently supports only PyTorch checkpoints")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(inputs)

        # RoboTwin resets torch with the accepted scene seed during setup. Resetting
        # here to that same seed reproduces the first torch.normal draw used by the
        # original policy rollout while making the contract explicit in the artifact.
        torch.manual_seed(int(noise_seed))
        noise = self._model.sample_noise(
            (1, self._model.config.action_horizon, self._model.config.action_dim),
            self._pytorch_device,
        )
        outputs = self._model.probe_solver_features(
            self._pytorch_device,
            observation,
            noise,
            trace_budgets=trace_budgets,
        )
        numpy_outputs = {
            key: np.asarray(value[0, ...].detach().cpu())
            for key, value in outputs.items()
        }
        numpy_outputs["trace_budgets"] = np.asarray(trace_budgets, dtype=np.int64)

        # Apply the exact inference output transform separately to every traced
        # endpoint. This records both normalized 32-D model space and the ordered
        # 14-D ALOHA chunk that would be sent to the evaluator.
        state = np.asarray(inputs["state"][0, ...].detach().cpu())
        for budget in trace_budgets:
            endpoint_key = f"trace_k{budget}_endpoint_raw"
            transformed = self._output_transform(
                {
                    "state": state.copy(),
                    "actions": numpy_outputs[endpoint_key].copy(),
                }
            )
            numpy_outputs[f"trace_k{budget}_actions"] = np.asarray(
                transformed["actions"]
            )
        return numpy_outputs

    def probe_conditional_candidate_consistency(
        self,
        obs: dict,
        *,
        noise_seed: int,
        trace_budgets: tuple[int, ...],
        condition_labels: tuple[str, ...],
        condition_prompts: tuple[str, ...],
        residual_noise_seeds: tuple[int, ...],
        residual_times: tuple[float, ...],
    ) -> dict[str, np.ndarray]:
        """Score fixed K candidates under paired prompt conditions without motion."""
        if not self._is_pytorch_model:
            raise NotImplementedError("conditional candidate probing requires PyTorch")
        if len(condition_labels) != len(condition_prompts) or not condition_labels:
            raise ValueError("condition labels and prompts must be non-empty and aligned")
        budgets = tuple(dict.fromkeys(int(budget) for budget in trace_budgets))
        if not budgets:
            raise ValueError("conditional candidate probing requires trace budgets")
        if not residual_noise_seeds or not residual_times:
            raise ValueError("residual noise seeds and times must be non-empty")

        condition_observations = []
        condition_probe_outputs = []
        candidate_origin_indices = []
        candidate_budgets = []
        candidate_endpoints = []
        candidate_actions = []
        for condition_index, prompt in enumerate(condition_prompts):
            condition_obs = jax.tree.map(lambda x: x, obs)
            condition_obs["prompt"] = str(prompt)
            condition_observations.append(condition_obs)
            probe_outputs = self.probe_solver_features(
                condition_obs,
                noise_seed=int(noise_seed),
                trace_budgets=budgets,
            )
            condition_probe_outputs.append(probe_outputs)
            for budget in budgets:
                candidate_origin_indices.append(condition_index)
                candidate_budgets.append(budget)
                candidate_endpoints.append(probe_outputs[f"trace_k{budget}_endpoint_raw"])
                candidate_actions.append(probe_outputs[f"trace_k{budget}_actions"])

        outputs = condition_probe_outputs[0]
        candidates = torch.from_numpy(np.stack(candidate_endpoints, axis=0)).to(
            self._pytorch_device
        )
        residual_noises = []
        for seed in residual_noise_seeds:
            torch.manual_seed(int(seed))
            residual_noises.append(
                self._model.sample_noise(
                    (self._model.config.action_horizon, self._model.config.action_dim),
                    self._pytorch_device,
                )
            )
        residual_noises_tensor = torch.stack(residual_noises, dim=0)
        residual_times_tensor = torch.tensor(
            residual_times,
            dtype=torch.float32,
            device=self._pytorch_device,
        )

        condition_losses = []
        token_ids = []
        token_masks = []
        transformed_state = None
        image_digest = hashlib.sha256()
        for camera_name in sorted(obs["images"]):
            camera = np.ascontiguousarray(np.asarray(obs["images"][camera_name]))
            image_digest.update(camera_name.encode("utf-8"))
            image_digest.update(camera.tobytes())

        for condition_obs in condition_observations:
            inputs = self._input_transform(condition_obs)
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                inputs,
            )
            observation = _model.Observation.from_dict(inputs)
            current_state = np.asarray(inputs["state"][0, ...].detach().cpu())
            if transformed_state is None:
                transformed_state = current_state
            elif not np.array_equal(transformed_state, current_state):
                raise ValueError("changing only the prompt changed transformed state")
            token_ids.append(
                np.asarray(observation.tokenized_prompt[0, ...].detach().cpu())
            )
            token_masks.append(
                np.asarray(observation.tokenized_prompt_mask[0, ...].detach().cpu())
            )
            loss = self._model.score_candidate_flow_residuals(
                self._pytorch_device,
                observation,
                candidates,
                residual_noises_tensor,
                residual_times_tensor,
            )
            condition_losses.append(np.asarray(loss.detach().cpu()))

        outputs.update(
            {
                "conditional_candidate_origin_indices": np.asarray(
                    candidate_origin_indices, dtype=np.int64
                ),
                "conditional_candidate_budgets": np.asarray(
                    candidate_budgets, dtype=np.int64
                ),
                "conditional_candidate_endpoint_raw": np.stack(
                    candidate_endpoints, axis=0
                ),
                "conditional_candidate_actions": np.stack(candidate_actions, axis=0),
                "conditional_condition_labels": np.asarray(condition_labels),
                "conditional_condition_prompts": np.asarray(condition_prompts),
                "conditional_original_prompt_index": np.asarray(0, dtype=np.int64),
                "conditional_residual_noise_seeds": np.asarray(
                    residual_noise_seeds, dtype=np.int64
                ),
                "conditional_residual_times": np.asarray(residual_times, dtype=np.float32),
                "conditional_residual_sq": np.stack(condition_losses, axis=0),
                "conditional_tokenized_prompt": np.stack(token_ids, axis=0),
                "conditional_tokenized_prompt_mask": np.stack(token_masks, axis=0),
                "conditional_raw_image_sha256": np.asarray(image_digest.hexdigest()),
            }
        )
        return outputs

    def probe_solver_feature_ensemble(
        self,
        obs: dict,
        *,
        noise_seed: int,
        ensemble_size: int,
        trace_budgets: tuple[int, ...] = (),
    ) -> dict[str, np.ndarray]:
        """Repeat a no-motion full trace at one observation over fixed noise seeds.

        This diagnostic separates statistics over stochastic action samples from
        statistics over the rows of one action chunk.  It intentionally uses the
        same single-sample path as ``probe_solver_features`` for every member so
        the ensemble does not introduce batch-size-dependent model numerics.
        """
        ensemble_size = int(ensemble_size)
        if ensemble_size <= 0:
            raise ValueError(f"ensemble_size must be positive, got {ensemble_size}")

        noise_seeds = np.arange(
            int(noise_seed),
            int(noise_seed) + ensemble_size,
            dtype=np.int64,
        )
        members = [
            self.probe_solver_features(
                obs,
                noise_seed=int(member_seed),
                trace_budgets=trace_budgets,
            )
            for member_seed in noise_seeds
        ]
        outputs = {
            key: np.stack([member[key] for member in members], axis=0)
            for key in members[0]
            if key != "trace_budgets"
        }
        outputs["trace_budgets"] = np.asarray(trace_budgets, dtype=np.int64)
        outputs["ensemble_noise_seeds"] = noise_seeds
        return outputs

    def infer_action_budgets(
        self,
        obs: dict,
        *,
        budgets: tuple[int, ...],
    ) -> dict[int, np.ndarray]:
        """Sample matched-noise action endpoints for several Euler budgets.

        This is a diagnostic rollout path.  Every endpoint shares the exact
        transformed observation, prefix computation, and sampled noise tensor.
        It deliberately reuses the eager full-field tracer so control and
        intervention rollouts do not mix eager and compiled solver numerics.
        """
        if not self._is_pytorch_model:
            raise NotImplementedError(
                "matched-budget action inference currently supports only PyTorch checkpoints"
            )
        budgets = tuple(dict.fromkeys(int(budget) for budget in budgets))
        if not budgets or any(budget <= 0 for budget in budgets):
            raise ValueError(f"budgets must be positive integers: {budgets}")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(inputs)
        noise = self._model.sample_noise(
            (1, self._model.config.action_horizon, self._model.config.action_dim),
            self._pytorch_device,
        )
        outputs = self._model.probe_solver_features(
            self._pytorch_device,
            observation,
            noise,
            trace_budgets=budgets,
        )

        state = np.asarray(inputs["state"][0, ...].detach().cpu())
        actions_by_budget = {}
        for budget in budgets:
            endpoint = np.asarray(
                outputs[f"trace_k{budget}_endpoint_raw"][0, ...].detach().cpu()
            )
            transformed = self._output_transform(
                {
                    "state": state.copy(),
                    "actions": endpoint,
                }
            )
            actions_by_budget[budget] = np.asarray(transformed["actions"])
        return actions_by_budget

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
