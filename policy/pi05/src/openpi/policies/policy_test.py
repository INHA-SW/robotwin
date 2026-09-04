from types import SimpleNamespace

import numpy as np
from openpi_client import action_chunk_broker
import pytest
import torch

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class _BudgetProbeModel:
    def __init__(self):
        self.config = SimpleNamespace(action_horizon=2, action_dim=3)
        self.include_diagnostics = []

    def sample_noise(self, shape, device):
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def probe_solver_features(
        self,
        device,
        observation,
        noise,
        *,
        trace_budgets,
        include_diagnostics,
    ):
        self.include_diagnostics.append(include_diagnostics)
        outputs = {
            "state": observation["state"],
            "noise": noise,
            "v0": torch.ones_like(noise),
        }
        if include_diagnostics:
            outputs["scalar_features"] = torch.ones(
                (1, 5), dtype=torch.float32, device=device
            )
        for budget in trace_budgets:
            outputs[f"trace_k{budget}_endpoint_raw"] = torch.full_like(
                noise, float(budget)
            )
        return outputs


def test_infer_action_budgets_persists_only_first_matched_trace(
    monkeypatch, tmp_path
):
    model = _BudgetProbeModel()
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._is_pytorch_model = True
    policy._pytorch_device = "cpu"
    policy._model = model
    policy._input_transform = lambda value: {
        **value,
        "prompt": np.asarray([1, 2], dtype=np.int64),
    }
    policy._output_transform = lambda value: {
        **value,
        "actions": value["actions"][:, :2],
    }
    monkeypatch.setattr(_policy._model.Observation, "from_dict", lambda value: value)
    trace_path = tmp_path / "first_matched_trace.npz"
    monkeypatch.setenv("PI05_MATCHED_BUDGET_TRACE_PATH", str(trace_path))
    observation = {
        "state": np.arange(3, dtype=np.float32),
        "images": {"cam_high": np.zeros((3, 4, 5), dtype=np.uint8)},
        "prompt": "test instruction",
    }

    first = policy.infer_action_budgets(
        observation, budgets=(1, 2), include_diagnostics=True
    )
    original = trace_path.read_bytes()
    second = policy.infer_action_budgets(
        observation, budgets=(2,), include_diagnostics=False
    )

    assert model.include_diagnostics == [True, False]
    assert first[1].shape == (2, 2)
    assert second[2].shape == (2, 2)
    assert trace_path.read_bytes() == original
    with np.load(trace_path, allow_pickle=False) as trace:
        assert trace["trace_budgets"].tolist() == [1, 2]
        assert trace["instruction"].item() == "test instruction"
        np.testing.assert_array_equal(trace["observation_state"], observation["state"])
        np.testing.assert_array_equal(
            trace["observation_cam_high"], observation["images"]["cam_high"]
        )
        np.testing.assert_array_equal(trace["trace_k2_actions"], first[2])


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
