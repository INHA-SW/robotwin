import json

import torch

from pi_model import PI0


def _uninitialized_model(tmp_path):
    model = PI0.__new__(PI0)
    model.reset_obsrvationwindows = lambda: None
    model._arm_rollout_replan_index = 99
    model.intervention_replan_index = 99
    model._episode_context = None
    model.rollout_record_root = str(tmp_path)
    model._default_intervention_records = ""
    model._default_inference_timing_records = ""
    model._default_matched_budget_trace_path = ""
    return model


def test_reset_model_routes_records_and_reseeds_torch(tmp_path):
    model = _uninitialized_model(tmp_path)
    first_context = {
        "episode_seed": 100002,
        "task_name": "beat_block_hammer",
        "task_config": "demo_clean",
        "episode_index": 0,
    }
    first = model.reset_model(first_context)
    first_noise = torch.rand(4)

    assert model._arm_rollout_replan_index == 0
    assert model.intervention_replan_index == 0
    assert model.intervention_records.endswith("intervention.jsonl")
    assert model.inference_timing_records.endswith("inference_timing.jsonl")
    assert model.matched_budget_trace_path.endswith("first_matched_trace.npz")
    first_root = tmp_path / "demo_clean" / "beat_block_hammer" / "seed_100002_episode_0"
    assert first["episode_record_root"] == str(first_root)
    assert json.loads((first_root / "episode_context.json").read_text()) == first_context

    second_context = {**first_context, "episode_index": 1}
    model.reset_model(second_context)
    second_noise = torch.rand(4)
    torch.testing.assert_close(first_noise, second_noise, rtol=0, atol=0)


def test_reset_model_rejects_unsafe_record_component(tmp_path):
    model = _uninitialized_model(tmp_path)
    context = {
        "episode_seed": 1,
        "task_name": "../escape",
        "task_config": "demo_clean",
        "episode_index": 0,
    }

    try:
        model.reset_model(context)
    except ValueError as error:
        assert "unsafe or missing episode context task_name" in str(error)
    else:
        raise AssertionError("unsafe task name was accepted")
