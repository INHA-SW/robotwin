#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import hashlib
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os
import time


def _parse_one_based_indices(spec, *, size, name):
    values = []
    for token in str(spec).replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"{name} range must be ascending: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    values = tuple(dict.fromkeys(values))
    if not values or min(values) < 1 or max(values) > size:
        raise ValueError(f"{name} must be unique 1-based indices in [1, {size}]: {spec}")
    return values

class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, num_inference_steps):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.pi0_step = int(pi0_step)
        self.num_inference_steps = int(num_inference_steps)
        self.action_intervention_mode = os.environ.get(
            "PI05_ACTION_INTERVENTION_MODE", ""
        ).strip()
        self.matched_budgets = tuple(
            dict.fromkeys(
                int(value)
                for value in os.environ.get(
                    "PI05_MATCHED_BUDGETS", "2,10"
                ).replace(",", " ").split()
            )
        )
        if not self.matched_budgets or any(budget <= 0 for budget in self.matched_budgets):
            raise ValueError(
                f"PI05_MATCHED_BUDGETS must contain positive integers: {self.matched_budgets}"
            )
        first_replan_only_text = os.environ.get(
            "PI05_FIRST_REPLAN_MATCHED_ONLY", "false"
        ).strip().lower()
        if first_replan_only_text not in {"true", "false"}:
            raise ValueError("PI05_FIRST_REPLAN_MATCHED_ONLY must be true or false")
        self.first_replan_matched_only = first_replan_only_text == "true"
        eager_single_budget_text = os.environ.get(
            "PI05_EAGER_SINGLE_BUDGET", "false"
        ).strip().lower()
        if eager_single_budget_text not in {"true", "false"}:
            raise ValueError(
                "PI05_EAGER_SINGLE_BUDGET must be true or false, got "
                f"{eager_single_budget_text!r}"
            )
        self.eager_single_budget = eager_single_budget_text == "true"
        valid_intervention_modes = {
            "",
            "k1_control",
            "k2_control",
            "k5_control",
            "k10_control",
            "k2_k10_alpha",
            "k2_k10_anti_alpha",
            "k2_k10_orthogonal_alpha",
            "k2_to_k10_mask",
            "k10_to_k2_mask",
        }
        if self.action_intervention_mode not in valid_intervention_modes:
            raise ValueError(
                "PI05_ACTION_INTERVENTION_MODE must be one of "
                f"{sorted(valid_intervention_modes)}, got {self.action_intervention_mode!r}"
            )
        control_budget = (
            int(self.action_intervention_mode[1:-len("_control")])
            if self.action_intervention_mode.endswith("_control")
            else None
        )
        if control_budget is not None and control_budget not in self.matched_budgets:
            raise ValueError(
                f"{self.action_intervention_mode} requires K{control_budget} in "
                f"PI05_MATCHED_BUDGETS={self.matched_budgets}"
            )
        if self.action_intervention_mode and (
            2 not in self.matched_budgets or 10 not in self.matched_budgets
        ):
            raise ValueError(
                "matched-budget intervention logging requires K2 and K10 candidates"
            )
        if self.eager_single_budget and self.action_intervention_mode:
            raise ValueError(
                "PI05_EAGER_SINGLE_BUDGET and PI05_ACTION_INTERVENTION_MODE "
                "cannot be enabled together"
            )
        intervention_alpha_text = os.environ.get(
            "PI05_INTERVENTION_ALPHA", ""
        ).strip()
        self.intervention_alpha = (
            float(intervention_alpha_text) if intervention_alpha_text else None
        )
        alpha_modes = {
            "k2_k10_alpha",
            "k2_k10_anti_alpha",
            "k2_k10_orthogonal_alpha",
        }
        if self.action_intervention_mode in alpha_modes:
            if self.intervention_alpha is None or not 0.0 <= self.intervention_alpha <= 1.0:
                raise ValueError(
                    "PI05_INTERVENTION_ALPHA must be in [0, 1] for alpha interventions"
                )
        elif self.intervention_alpha is not None:
            raise ValueError(
                "PI05_INTERVENTION_ALPHA is only valid with an alpha intervention"
            )
        self.intervention_horizons = _parse_one_based_indices(
            os.environ.get("PI05_INTERVENTION_HORIZONS_1BASED", "8-16,25-32"),
            size=32,
            name="PI05_INTERVENTION_HORIZONS_1BASED",
        )
        self.intervention_actions = _parse_one_based_indices(
            os.environ.get("PI05_INTERVENTION_ACTIONS_1BASED", "9-11,13"),
            size=14,
            name="PI05_INTERVENTION_ACTIONS_1BASED",
        )
        self.intervention_records = os.environ.get(
            "PI05_INTERVENTION_RECORDS", ""
        ).strip()
        self.inference_timing_records = os.environ.get(
            "PI05_INFERENCE_TIMING_RECORDS", ""
        ).strip()
        self.intervention_replan_index = 0
        terminal_jump_text = os.environ.get("PI05_TERMINAL_JUMP_TIME", "").strip()
        self.terminal_jump_time = (
            float(terminal_jump_text) if terminal_jump_text else None
        )
        if self.pi0_step <= 0:
            raise ValueError(f"pi0_step must be positive, got {self.pi0_step}")
        if self.num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {self.num_inference_steps}"
            )
        if self.terminal_jump_time is not None:
            if self.num_inference_steps < 2:
                raise ValueError("terminal-jump sampling requires at least 2 NFEs")
            if not 0.0 < self.terminal_jump_time < 1.0:
                raise ValueError(
                    "PI05_TERMINAL_JUMP_TIME must be strictly between 0 and 1, "
                    f"got {self.terminal_jump_time}"
                )

        default_checkpoint_dir = os.path.join(
            "policy",
            "pi05",
            "checkpoints",
            self.train_config_name,
            self.model_name,
            str(self.checkpoint_id),
        )
        self.checkpoint_dir = os.path.abspath(
            os.environ.get("PI05_CHECKPOINT_DIR", default_checkpoint_dir)
        )
        specified_path = os.path.join(self.checkpoint_dir, "assets")
        entries = os.listdir(specified_path)
        if len(entries) != 1:
            raise ValueError(
                f"Expected exactly one normalization asset under {specified_path}, "
                f"found {entries}"
            )
        assets_id = entries[0]

        config = _config.get_config(self.train_config_name)
        sample_kwargs = {"num_steps": self.num_inference_steps}
        if self.terminal_jump_time is not None:
            sample_kwargs["terminal_jump_time"] = self.terminal_jump_time
        self.policy = _policy_config.create_trained_policy(
            config,
            self.checkpoint_dir,
            robotwin_repo_id=assets_id,
            sample_kwargs=sample_kwargs,
            )
        print(
            "loading model success! "
            f"action_execution_steps={self.pi0_step} "
            f"flow_denoising_steps={self.num_inference_steps} "
            f"terminal_jump_time={self.terminal_jump_time} "
            f"eager_single_budget={self.eager_single_budget} "
            f"intervention_alpha={self.intervention_alpha} "
            f"action_intervention_mode={self.action_intervention_mode or 'none'}"
        )
        self.img_size = (224, 224)
        self.observation_window = None
        self._arm_rollout_replan_index = 0
        self._episode_context = None

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        if not self.action_intervention_mode:
            if self.eager_single_budget:
                return self.policy.infer_action_budgets(
                    self.observation_window,
                    budgets=(self.num_inference_steps,),
                )[self.num_inference_steps]
            return self.policy.infer(self.observation_window)["actions"]

        if (
            self.first_replan_matched_only
            and self.intervention_replan_index > 0
            and self.action_intervention_mode.endswith("_control")
        ):
            control_budget = int(self.action_intervention_mode[1:-len("_control")])
            inference_started = time.monotonic()
            selected = self.policy.infer_action_budgets(
                self.observation_window,
                budgets=(control_budget,),
                include_diagnostics=False,
            )[control_budget]
            self._append_inference_timing(
                phase="selected_budget_only",
                computed_budgets=(control_budget,),
                selected_budget=control_budget,
                wall_seconds=time.monotonic() - inference_started,
            )
            self.intervention_replan_index += 1
            return np.asarray(selected)

        inference_started = time.monotonic()
        actions_by_budget = self.policy.infer_action_budgets(
            self.observation_window,
            budgets=self.matched_budgets,
        )
        inference_wall_seconds = time.monotonic() - inference_started
        candidates = {
            budget: np.asarray(actions_by_budget[budget])
            for budget in self.matched_budgets
        }
        invalid_shapes = {
            budget: actions.shape
            for budget, actions in candidates.items()
            if actions.shape != (32, 14)
        }
        if invalid_shapes:
            raise ValueError(f"expected matched (32, 14) chunks, got {invalid_shapes}")
        k2_actions = candidates[2]
        k10_actions = candidates[10]

        selected = k10_actions.copy()
        source_budget = 2
        base_budget = 10
        if self.action_intervention_mode.endswith("_control"):
            control_budget = int(self.action_intervention_mode[1:-len("_control")])
            selected = candidates[control_budget].copy()
            source_budget = control_budget
            base_budget = control_budget
        k10_direction = k10_actions - k2_actions
        applied_direction = np.zeros_like(k10_direction)
        if self.action_intervention_mode == "k2_k10_alpha":
            applied_direction = k10_direction
            selected = k2_actions + self.intervention_alpha * applied_direction
            source_budget = 10
            base_budget = 2
        elif self.action_intervention_mode == "k2_k10_anti_alpha":
            applied_direction = -k10_direction
            selected = k2_actions + self.intervention_alpha * applied_direction
            source_budget = 10
            base_budget = 2
        elif self.action_intervention_mode == "k2_k10_orthogonal_alpha":
            # Rotate only along the horizon so joint/gripper channels retain their
            # physical units, then remove the K2->K10 component and restore its RMS.
            candidate = np.roll(k10_direction, shift=1, axis=0)
            denominator = float(np.sum(np.square(k10_direction)))
            if denominator <= np.finfo(np.float64).eps:
                raise ValueError("cannot construct a sham direction from zero K2-K10 delta")
            candidate -= (
                float(np.sum(candidate * k10_direction)) / denominator
            ) * k10_direction
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm <= np.finfo(np.float64).eps:
                raise ValueError("failed to construct a nonzero orthogonal sham direction")
            applied_direction = candidate * (
                float(np.sqrt(denominator)) / candidate_norm
            )
            selected = k2_actions + self.intervention_alpha * applied_direction
            source_budget = 10
            base_budget = 2
        elif self.action_intervention_mode == "k2_to_k10_mask":
            horizon_indices = np.asarray(self.intervention_horizons, dtype=np.int64) - 1
            action_indices = np.asarray(self.intervention_actions, dtype=np.int64) - 1
            selected[np.ix_(horizon_indices, action_indices)] = k2_actions[
                np.ix_(horizon_indices, action_indices)
            ]
        elif self.action_intervention_mode == "k10_to_k2_mask":
            selected = k2_actions.copy()
            source_budget = 10
            base_budget = 2
            horizon_indices = np.asarray(self.intervention_horizons, dtype=np.int64) - 1
            action_indices = np.asarray(self.intervention_actions, dtype=np.int64) - 1
            selected[np.ix_(horizon_indices, action_indices)] = k10_actions[
                np.ix_(horizon_indices, action_indices)
            ]

        delta = k2_actions - k10_actions
        horizon_indices = np.asarray(self.intervention_horizons, dtype=np.int64) - 1
        action_indices = np.asarray(self.intervention_actions, dtype=np.int64) - 1
        masked_delta = delta[np.ix_(horizon_indices, action_indices)]
        k10_direction_norm = float(np.linalg.norm(k10_direction))
        applied_direction_norm = float(np.linalg.norm(applied_direction))
        direction_cosine = (
            float(np.sum(k10_direction * applied_direction))
            / (k10_direction_norm * applied_direction_norm)
            if k10_direction_norm > 0.0 and applied_direction_norm > 0.0
            else None
        )
        record = {
            "schema_version": 1,
            "record_kind": "matched_budget_action_intervention",
            "run_id": os.environ.get("ARM_EVAL_RUN_ID"),
            "mode": self.action_intervention_mode,
            "replan_index": self.intervention_replan_index,
            "source_budget": source_budget,
            "base_budget": base_budget,
            "alpha": self.intervention_alpha,
            "horizon_indices_1based": list(self.intervention_horizons),
            "action_indices_1based": list(self.intervention_actions),
            "action_schema": (
                "left_j1..j6,left_gripper,right_j1..j6,right_gripper"
            ),
            "candidate_budgets": list(self.matched_budgets),
            "k2_k10_rms": float(np.sqrt(np.mean(np.square(delta)))),
            "masked_k2_k10_rms": float(
                np.sqrt(np.mean(np.square(masked_delta)))
            ),
            "masked_k2_k10_max_abs": float(np.max(np.abs(masked_delta))),
            "applied_direction_rms": float(
                np.sqrt(np.mean(np.square(applied_direction)))
            ),
            "applied_direction_cosine_to_k10": direction_cosine,
            "selected_k2_rms": float(
                np.sqrt(np.mean(np.square(selected - k2_actions)))
            ),
            "k2_sha256": hashlib.sha256(
                np.ascontiguousarray(k2_actions).tobytes()
            ).hexdigest(),
            "k10_sha256": hashlib.sha256(
                np.ascontiguousarray(k10_actions).tobytes()
            ).hexdigest(),
            "selected_sha256": hashlib.sha256(
                np.ascontiguousarray(selected).tobytes()
            ).hexdigest(),
        }
        for budget, actions in candidates.items():
            record[f"k{budget}_sha256"] = hashlib.sha256(
                np.ascontiguousarray(actions).tobytes()
            ).hexdigest()
        if self.intervention_records:
            with open(self.intervention_records, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
        selected_budget = (
            int(self.action_intervention_mode[1:-len("_control")])
            if self.action_intervention_mode.endswith("_control")
            else None
        )
        self._append_inference_timing(
            phase="matched_candidates_before_first_motion",
            computed_budgets=self.matched_budgets,
            selected_budget=selected_budget,
            wall_seconds=inference_wall_seconds,
        )
        print(
            "[arm] matched-budget intervention "
            f"mode={self.action_intervention_mode} "
            f"replan={self.intervention_replan_index} "
            f"full_rms={record['k2_k10_rms']:.6f} "
            f"masked_rms={record['masked_k2_k10_rms']:.6f}",
            flush=True,
        )
        self.intervention_replan_index += 1
        return selected

    def _set_transport_observation(self, payload):
        """Validate and install one simulator observation received over the socket."""
        if not isinstance(payload, dict):
            raise TypeError(
                f"observation payload must be a dict, got {type(payload).__name__}"
            )
        images = payload.get("images")
        state = payload.get("state")
        instruction = payload.get("instruction")
        if not isinstance(images, dict):
            raise ValueError("act payload requires an images mapping")
        required = ("head_camera", "right_camera", "left_camera")
        missing = [name for name in required if name not in images]
        if missing:
            raise ValueError(f"act payload is missing images: {missing}")
        if state is None or not str(instruction or "").strip():
            raise ValueError("act payload requires state and non-empty instruction")

        if self.observation_window is None or instruction != self.instruction:
            self.set_language(str(instruction))
        self.update_observation_window(
            [images[name] for name in required],
            np.asarray(state),
        )
        return self.observation_window

    def act(self, payload):
        """Run one policy replan from a transport-safe observation payload."""
        self._set_transport_observation(payload)
        actions = np.asarray(self.get_action()[: self.pi0_step], dtype=np.float32)
        self._arm_rollout_replan_index += 1
        return actions

    def probe(self, payload):
        """Capture matched-NFE traces for one transported observation without motion."""
        observation = self._set_transport_observation(payload)
        if "noise_seed" not in payload:
            raise ValueError("probe payload requires an explicit noise_seed")
        budgets = tuple(
            dict.fromkeys(
                int(value)
                for value in payload.get("trace_budgets", self.matched_budgets)
            )
        )
        if not budgets or any(budget <= 0 for budget in budgets):
            raise ValueError(f"probe trace_budgets must be positive: {budgets}")

        features = self.policy.probe_solver_features(
            observation,
            noise_seed=int(payload["noise_seed"]),
            trace_budgets=budgets,
        )
        features.update(
            {
                "observation_state": np.asarray(observation["state"]),
                "observation_cam_high": np.asarray(
                    observation["images"]["cam_high"]
                ),
                "observation_cam_left_wrist": np.asarray(
                    observation["images"]["cam_left_wrist"]
                ),
                "observation_cam_right_wrist": np.asarray(
                    observation["images"]["cam_right_wrist"]
                ),
                "instruction": str(observation["prompt"]),
            }
        )
        return features

    def health(self):
        """Return the loaded policy contract for local transport preflights."""
        return {
            "schema_version": 1,
            "status": "ready",
            "train_config_name": self.train_config_name,
            "model_name": self.model_name,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_dir": self.checkpoint_dir,
            "action_execution_steps": self.pi0_step,
            "num_inference_steps": self.num_inference_steps,
            "matched_budgets": list(self.matched_budgets),
        }

    def reset_model(self, episode_context=None):
        """Reset recurrent state while retaining paired episode metadata."""
        self.reset_obsrvationwindows()
        self._arm_rollout_replan_index = 0
        self._episode_context = episode_context
        return {
            "reset": True,
            "episode_context": episode_context,
        }

    def _append_inference_timing(
        self,
        *,
        phase,
        computed_budgets,
        selected_budget,
        wall_seconds,
    ):
        if not self.inference_timing_records:
            return
        computed_budgets = tuple(int(value) for value in computed_budgets)
        diagnostics_enabled = phase == "matched_candidates_before_first_motion"
        field_evaluations = 1 + int(diagnostics_enabled) + sum(
            0
            if diagnostics_enabled and budget == 2
            else max(budget - 1, 0)
            for budget in computed_budgets
        )
        record = {
            "schema_version": 1,
            "record_kind": "pi05_cross_nfe_inference_timing",
            "run_id": os.environ.get("ARM_EVAL_RUN_ID"),
            "mode": self.action_intervention_mode,
            "replan_index": self.intervention_replan_index,
            "phase": str(phase),
            "computed_budgets": list(computed_budgets),
            "selected_budget": (
                int(selected_budget) if selected_budget is not None else None
            ),
            "wall_seconds": float(wall_seconds),
            "timing_scope": "eager_matched_budget_policy_call_only",
            "diagnostics_enabled": diagnostics_enabled,
            "nominal_nfe_sum": int(sum(computed_budgets)),
            "actual_field_evaluations": int(field_evaluations),
        }
        with open(self.inference_timing_records, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def get_solver_probe(self, noise_seed):
        assert self.observation_window is not None, "update observation_window first!"
        trace_text = os.environ.get("ROBOTWIN_SOLVER_TRACE_BUDGETS", "").strip()
        trace_budgets = tuple(
            int(value)
            for value in trace_text.replace(",", " ").split()
        )
        ensemble_size = int(
            os.environ.get("ROBOTWIN_SOLVER_TRACE_ENSEMBLE_SIZE", "1")
        )
        conditional_prompt_path = os.environ.get(
            "PI05_CONDITIONAL_PROMPTS_PATH", ""
        ).strip()
        if conditional_prompt_path:
            if ensemble_size != 1:
                raise ValueError(
                    "conditional candidate consistency and trace ensemble are mutually exclusive"
                )
            with open(conditional_prompt_path, encoding="utf-8") as handle:
                condition_map = json.load(handle)
            if not isinstance(condition_map, dict) or not condition_map:
                raise ValueError("conditional prompts must be a non-empty JSON object")
            labels = tuple(str(label) for label in condition_map)
            prompts = tuple(str(prompt) for prompt in condition_map.values())
            if labels[0] != "true" or prompts[0] != str(self.instruction):
                raise ValueError(
                    "the first conditional prompt must be label 'true' and exactly match the manifest instruction"
                )
            residual_times = tuple(
                float(value)
                for value in os.environ.get(
                    "PI05_CONDITIONAL_RESIDUAL_TIMES",
                    "0.25075,0.52051,0.73139,0.91486",
                ).replace(" ", "").split(",")
                if value
            )
            residual_count = int(
                os.environ.get("PI05_CONDITIONAL_RESIDUAL_NOISE_COUNT", "2")
            )
            residual_offset = int(
                os.environ.get("PI05_CONDITIONAL_RESIDUAL_NOISE_OFFSET", "1000000")
            )
            residual_noise_seeds = tuple(
                int(noise_seed) + residual_offset + index
                for index in range(residual_count)
            )
            return self.policy.probe_conditional_candidate_consistency(
                self.observation_window,
                noise_seed=int(noise_seed),
                trace_budgets=trace_budgets,
                condition_labels=labels,
                condition_prompts=prompts,
                residual_noise_seeds=residual_noise_seeds,
                residual_times=residual_times,
            )
        if ensemble_size > 1:
            return self.policy.probe_solver_feature_ensemble(
                self.observation_window,
                noise_seed=int(noise_seed),
                ensemble_size=ensemble_size,
                trace_budgets=trace_budgets,
            )
        return self.policy.probe_solver_features(
            self.observation_window,
            noise_seed=int(noise_seed),
            trace_budgets=trace_budgets,
        )

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.intervention_replan_index = 0
        print("successfully unset obs and language intruction")
