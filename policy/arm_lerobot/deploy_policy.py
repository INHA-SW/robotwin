from __future__ import annotations

import copy
import contextlib
import json
import os
import signal
import sys
import faulthandler
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RENAME_MAP = {
    "observation.images.head_camera": "observation.images.cam_high",
    "observation.images.left_camera": "observation.images.cam_left_wrist",
    "observation.images.right_camera": "observation.images.cam_right_wrist",
}

ROBOTWIN_DUAL_ALOHA_GRIPPER_INDICES = (6, 13)


def _rtc_prefix_attention_horizon(
    prediction_horizon: int,
    execution_horizon: int,
) -> int:
    """Return the old/new action overlap used by RTC Eq. (5)."""

    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    if not 0 < execution_horizon <= prediction_horizon:
        raise ValueError(
            "execution_horizon must satisfy 0 < s <= prediction_horizon"
        )
    return prediction_horizon - execution_horizon


def _time_aligned_gaussian_coupling(
    previous_noise: Any,
    fresh_noise: Any,
    *,
    origin_shift_steps: int,
    rho: float,
):
    """Shift old flow noise in absolute time and correlate only its overlap."""

    if previous_noise.shape != fresh_noise.shape:
        raise ValueError(
            "C2 noise tensors must have identical shapes: "
            f"{tuple(previous_noise.shape)} != {tuple(fresh_noise.shape)}"
        )
    if previous_noise.ndim != 3:
        raise ValueError(f"Expected C2 noise [B,H,D], got {tuple(previous_noise.shape)}")
    horizon = int(previous_noise.shape[1])
    if not 0 <= origin_shift_steps <= horizon:
        raise ValueError((origin_shift_steps, horizon))
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"C2 rho must be in [0,1], got {rho}")

    coupled = fresh_noise.clone()
    overlap = horizon - origin_shift_steps
    if overlap:
        fresh_scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
        coupled[:, :overlap] = (
            rho * previous_noise[:, origin_shift_steps:]
            + fresh_scale * fresh_noise[:, :overlap]
        )
    return coupled


def _absolute_qpos_future_state(
    actual_request_state: np.ndarray,
    queued_actions: np.ndarray,
    handoff_steps: int,
) -> tuple[np.ndarray, int]:
    """Perfect-target absolute-qpos handoff state used by M3-FVE."""

    actual = np.asarray(actual_request_state, dtype=np.float32).reshape(-1)
    queued = np.asarray(queued_actions, dtype=np.float32)
    if queued.ndim != 2 or queued.shape[1] != actual.shape[0]:
        raise ValueError(
            f"Expected queued absolute qpos [T,{actual.shape[0]}], got {queued.shape}"
        )
    if handoff_steps < 0:
        raise ValueError("handoff_steps must be non-negative")
    used_steps = min(handoff_steps, len(queued))
    if used_steps == 0:
        return actual.copy(), 0
    return queued[used_steps - 1].copy(), used_steps


def _absolute_qpos_predicted_states(
    anchor_state: np.ndarray,
    queued_actions: np.ndarray,
) -> np.ndarray:
    """States before each absolute-qpos command under perfect target tracking."""

    anchor = np.asarray(anchor_state, dtype=np.float32).reshape(1, -1)
    queued = np.asarray(queued_actions, dtype=np.float32)
    if queued.ndim != 2 or queued.shape[1] != anchor.shape[1]:
        raise ValueError(
            f"Expected queued absolute qpos [T,{anchor.shape[1]}], got {queued.shape}"
        )
    return np.concatenate((anchor, queued), axis=0)


def _apply_absolute_qpos_sit(
    processed_actions: np.ndarray,
    *,
    crop_steps: int,
    predicted_old_states: np.ndarray,
    measured_request_state: np.ndarray,
    action_lower: np.ndarray,
    action_upper: np.ndarray,
    gripper_indices: tuple[int, ...] = ROBOTWIN_DUAL_ALOHA_GRIPPER_INDICES,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Transport request-time qpos innovation to the first handoff action."""

    actions = np.asarray(processed_actions, dtype=np.float32).copy()
    old_states = np.asarray(predicted_old_states, dtype=np.float32)
    measured = np.asarray(measured_request_state, dtype=np.float32).reshape(-1)
    lower = np.asarray(action_lower, dtype=np.float32).reshape(-1)
    upper = np.asarray(action_upper, dtype=np.float32).reshape(-1)
    if actions.ndim != 2 or actions.shape[1] != measured.shape[0]:
        raise ValueError(f"Invalid M3-SIT action shape {actions.shape}")
    if old_states.ndim != 2 or old_states.shape[1] != measured.shape[0]:
        raise ValueError(f"Invalid M3-SIT old-state shape {old_states.shape}")
    if not 0 <= crop_steps < len(actions) or crop_steps >= len(old_states):
        raise ValueError((crop_steps, len(actions), len(old_states)))
    if lower.shape != measured.shape or upper.shape != measured.shape:
        raise ValueError("M3-SIT action bounds do not match the action dimension")

    grippers = set(gripper_indices)
    arm_indices = np.asarray(
        [index for index in range(measured.shape[0]) if index not in grippers],
        dtype=np.int64,
    )
    correction = measured - old_states[0]
    base = actions[crop_steps, arm_indices]
    raw = base + correction[arm_indices]
    # q01/q99 bounds limit only additional residual damage. If the base policy
    # is already outside a quantile bound, preserve that value as the local
    # envelope so zero innovation is an exact identity transform.
    effective_lower = np.minimum(lower[arm_indices], base)
    effective_upper = np.maximum(upper[arm_indices], base)
    saturated = int(
        np.count_nonzero(
            (raw < effective_lower) | (raw > effective_upper)
        )
    )
    actions[crop_steps, arm_indices] = np.clip(
        raw,
        effective_lower,
        effective_upper,
    )
    return actions, correction[arm_indices].copy(), saturated


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_json_dict(value: Any, default: dict[str, str]) -> dict[str, str]:
    if value is None or value == "":
        return dict(default)
    if isinstance(value, dict):
        return dict(value)
    return json.loads(str(value))


def _maybe_prepend_lerobot_src(path: str | None) -> None:
    if not path:
        return
    path = os.path.abspath(os.path.expanduser(path))
    if path not in sys.path:
        sys.path.insert(0, path)


def _log(message: str) -> None:
    print(f"[arm_lerobot] {message}", flush=True)


def _debug_enabled() -> bool:
    return _parse_bool(os.environ.get("ARM_ROBOTWIN_DEBUG"))


def _debug(message: str) -> None:
    if _debug_enabled():
        print(f"[arm_lerobot-debug {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _enable_stack_dump() -> None:
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    except Exception:
        pass


class LeRobotPolicyAdapter:
    """Causal chunk controller for LeRobot π0.5 in upstream RoboTwin."""

    def __init__(self, usr_args: dict[str, Any]):
        _enable_stack_dump()
        _log("initializing adapter")
        _maybe_prepend_lerobot_src(usr_args.get("lerobot_src"))

        _log("importing torch and lerobot helpers")
        import torch
        from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule
        from lerobot.policies import (
            get_policy_class,
            make_pre_post_processors,
            prepare_observation_for_inference,
        )
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        policy_path = usr_args.get("policy_path")
        if not policy_path:
            raise ValueError("arm_lerobot requires --policy_path=<LeRobot pretrained_model dir>")

        self.policy_path = Path(str(policy_path)).expanduser()
        self.device = torch.device(str(usr_args.get("device") or "cuda"))
        self.use_amp = _parse_bool(usr_args.get("use_amp"))
        self.action_dim = _parse_int(usr_args.get("action_dim"), 14)
        self.robot_type = str(usr_args.get("robot_type") or "aloha-agilex")
        self.rename_map = _parse_json_dict(usr_args.get("rename_map"), DEFAULT_RENAME_MAP)
        self.realtime_method = str(usr_args.get("realtime_method") or "sync")
        if self.realtime_method not in {
            "sync",
            "async",
            "c2",
            "m1r1",
            "m3_fve",
            "m3_sit",
            "rtc",
        }:
            raise ValueError(
                "realtime_method must be one of sync, async, c2, m1r1, "
                "m3_fve, m3_sit, rtc; "
                f"got {self.realtime_method!r}"
            )
        self.fixed_delay_steps = _parse_int(usr_args.get("fixed_delay_steps"), 4)
        self.execution_horizon = _parse_int(usr_args.get("execution_horizon"), 5)
        self.refresh_after_step = _parse_int(usr_args.get("refresh_after_step"), 5)
        self.realtime_seed = _parse_int(usr_args.get("realtime_seed"), 0)
        self.realtime_episode_offset = _parse_int(
            usr_args.get("realtime_episode_offset"), 0
        )
        if self.realtime_episode_offset < 0:
            raise ValueError(
                "realtime_episode_offset must be non-negative, got "
                f"{self.realtime_episode_offset}"
            )
        self.c2_rho = float(usr_args.get("c2_rho") or 0.75)
        if not 0.0 <= self.c2_rho <= 1.0:
            raise ValueError(f"c2_rho must be in [0,1], got {self.c2_rho}")
        self.rtc_max_guidance_weight = float(
            usr_args.get("rtc_max_guidance_weight") or 10.0
        )
        events_path_text = str(
            usr_args.get("realtime_events_path")
            or os.environ.get("ROBOTWIN_REALTIME_EVENTS")
            or ""
        )
        self.events_path = Path(events_path_text) if events_path_text else None
        self._torch = torch
        self._prepare_observation_for_inference = prepare_observation_for_inference

        _log(f"loading policy config from {self.policy_path}")
        policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        policy_cfg.pretrained_path = self.policy_path
        policy_cfg.device = str(self.device)
        policy_cfg.use_amp = self.use_amp

        num_inference_steps = usr_args.get("num_inference_steps")
        if num_inference_steps not in (None, "") and hasattr(policy_cfg, "num_inference_steps"):
            policy_cfg.num_inference_steps = int(num_inference_steps)
        prediction_horizon = int(policy_cfg.chunk_size)
        rtc_prefix_attention_horizon = _rtc_prefix_attention_horizon(
            prediction_horizon,
            self.execution_horizon,
        )
        if self.realtime_method == "rtc":
            policy_cfg.rtc_config = RTCConfig(
                enabled=True,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
                # LeRobot's RTCConfig inherited the name `execution_horizon`,
                # but RTCProcessor uses this value as Eq. (5)'s exclusive mask
                # end.  That is the H-s old/new overlap, not the replan period s.
                execution_horizon=rtc_prefix_attention_horizon,
                max_guidance_weight=self.rtc_max_guidance_weight,
            )
        else:
            policy_cfg.rtc_config = None

        _log(f"loading policy weights type={policy_cfg.type} device={self.device}")
        policy_cls = get_policy_class(policy_cfg.type)
        self.policy = policy_cls.from_pretrained(self.policy_path, config=policy_cfg)
        _log("moving policy to device")
        self.policy.to(self.device)
        self.policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
            "rename_observations_processor": {"rename_map": self.rename_map},
        }
        _log("building pre/post processors")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(self.policy_path),
            preprocessor_overrides=preprocessor_overrides,
        )
        _log("pre/post processors ready")
        self.prediction_horizon = prediction_horizon
        self.rtc_prefix_attention_horizon = rtc_prefix_attention_horizon
        self.max_action_dim = int(policy_cfg.max_action_dim)
        self.num_inference_steps = int(policy_cfg.num_inference_steps)
        if not 0 <= self.fixed_delay_steps <= self.execution_horizon:
            raise ValueError(
                "fixed_delay_steps must satisfy 0 <= d <= execution_horizon"
            )
        if self.execution_horizon > self.prediction_horizon - self.fixed_delay_steps:
            raise ValueError(
                "execution_horizon must satisfy e <= prediction_horizon - d"
            )
        if (
            self.realtime_method == "m1r1"
            and not 0 < self.refresh_after_step < self.num_inference_steps
        ):
            raise ValueError(
                "refresh_after_step must be strictly inside the flow solve"
            )
        if self.realtime_method in {"m3_fve", "m3_sit"} and self.action_dim != 14:
            raise ValueError(
                "RoboTwin M3 is defined for the dual-Aloha 14-D absolute-qpos layout"
            )

        self._action_lower: np.ndarray | None = None
        self._action_upper: np.ndarray | None = None
        if self.realtime_method == "m3_sit":
            from safetensors.torch import load_file

            stats_path = (
                self.policy_path
                / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
            )
            if not stats_path.is_file():
                raise FileNotFoundError(
                    "M3-SIT requires checkpoint action quantiles: " f"{stats_path}"
                )
            stats = load_file(stats_path, device="cpu")
            try:
                lower = stats["action.q01"]
                upper = stats["action.q99"]
            except KeyError as exc:
                raise KeyError(
                    "M3-SIT requires action.q01/action.q99 in postprocessor stats"
                ) from exc
            self._action_lower = lower[: self.action_dim].numpy().astype(np.float32)
            self._action_upper = upper[: self.action_dim].numpy().astype(np.float32)

        self._noise_generator = self._torch.Generator(device=self.device)

        self._condition = threading.Condition()
        self._worker: threading.Thread | None = None
        self._completed: dict[str, Any] | None = None
        self._worker_error: BaseException | None = None
        self._cancelled = False
        self._episode_generation = 0
        self._episode_index = self.realtime_episode_offset - 1
        self._initialized = False
        self._control_step = 0
        self._next_request_step = 0
        self._observation_history: dict[int, dict[str, Any]] = {}
        self._original_queue: deque[Any] = deque()
        self._processed_queue: deque[np.ndarray] = deque()
        self._predicted_qpos_states: deque[np.ndarray] = deque()
        self._previous_initial_noise: Any | None = None
        self._previous_noise_origin_step: int | None = None
        print(
            "ARM LeRobot policy adapter loaded: "
            f"path={self.policy_path}, type={policy_cfg.type}, "
            f"device={self.device}, steps={self.num_inference_steps}, "
            f"method={self.realtime_method}, d={self.fixed_delay_steps}, "
            f"e={self.execution_horizon}, H={self.prediction_horizon}, "
            f"rtc_overlap={self.rtc_prefix_attention_horizon}, "
            f"episode_offset={self.realtime_episode_offset}",
            flush=True,
        )

    def reset_model(self) -> None:
        worker = None
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join()

        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()
        self._episode_index += 1
        episode_seed = self.realtime_seed + self._episode_index
        self._torch.manual_seed(episode_seed)
        self._noise_generator.manual_seed(episode_seed)
        if self.device.type == "cuda":
            self._torch.cuda.manual_seed_all(episode_seed)
        with self._condition:
            self._episode_generation += 1
            self._worker = None
            self._completed = None
            self._worker_error = None
            self._cancelled = False
            self._initialized = False
            self._control_step = 0
            self._next_request_step = 0
            self._observation_history.clear()
            self._original_queue.clear()
            self._processed_queue.clear()
            self._predicted_qpos_states.clear()
            self._previous_initial_noise = None
            self._previous_noise_origin_step = None
        return None

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self.events_path is None:
            return
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "created_at": datetime.now().astimezone().isoformat(),
            "episode_index": self._episode_index,
            "method": self.realtime_method,
            "fixed_delay_steps": self.fixed_delay_steps,
            "execution_horizon": self.execution_horizon,
            "prediction_horizon": self.prediction_horizon,
            "rtc_prefix_attention_horizon": (
                self.rtc_prefix_attention_horizon
                if self.realtime_method == "rtc"
                else None
            ),
            "num_inference_steps": self.num_inference_steps,
            **event,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _prepare_batch(self, payload: dict[str, Any]):
        observation = payload["observation"]
        task = payload.get("instruction") or payload.get("task") or ""
        batch = self._prepare_observation_for_inference(
            copy.copy(observation),
            self.device,
            task,
            self.robot_type,
        )
        return self.preprocessor(batch)

    def _fresh_initial_noise(self):
        return self._torch.randn(
            (1, self.prediction_horizon, self.max_action_dim),
            device=self.device,
            dtype=self._torch.float32,
            generator=self._noise_generator,
        )

    def _initial_noise_for_request(
        self,
        *,
        request_step: int,
        warm_start: bool,
    ) -> tuple[Any, int | None]:
        fresh = self._fresh_initial_noise()
        if (
            self.realtime_method != "c2"
            or warm_start
            or self._previous_initial_noise is None
            or self._previous_noise_origin_step is None
        ):
            return fresh, None
        origin_shift = request_step - self._previous_noise_origin_step
        if not 0 <= origin_shift <= self.prediction_horizon:
            return fresh, None
        return (
            _time_aligned_gaussian_coupling(
                self._previous_initial_noise,
                fresh,
                origin_shift_steps=origin_shift,
                rho=self.c2_rho,
            ),
            origin_shift,
        )

    def _wait_for_refresh_batch(
        self,
        *,
        target_step: int,
        episode_generation: int,
    ):
        with self._condition:
            while (
                target_step not in self._observation_history
                and not self._cancelled
                and episode_generation == self._episode_generation
            ):
                self._condition.wait(timeout=0.1)
            if self._cancelled or episode_generation != self._episode_generation:
                return None
            payload = copy.deepcopy(self._observation_history[target_step])
        return self._prepare_batch(payload), target_step

    def _predict_chunk(
        self,
        payload: dict[str, Any],
        *,
        request_step: int,
        warm_start: bool,
        previous_chunk: Any | None,
        previous_processed: np.ndarray,
        predicted_old_states: np.ndarray | None,
        episode_generation: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        request_state = np.asarray(
            payload["observation"]["observation.state"], dtype=np.float32
        ).reshape(-1)
        inference_payload = copy.deepcopy(payload)
        m3_fve_state = None
        m3_fve_forward_steps = 0
        if self.realtime_method == "m3_fve" and not warm_start:
            m3_fve_state, m3_fve_forward_steps = _absolute_qpos_future_state(
                request_state,
                previous_processed,
                self.fixed_delay_steps,
            )
            inference_payload["observation"]["observation.state"] = m3_fve_state
        batch = self._prepare_batch(inference_payload)
        diagnostics: list[dict[str, Any]] = []
        initial_noise, coupling_origin_shift = self._initial_noise_for_request(
            request_step=request_step,
            warm_start=warm_start,
        )
        kwargs: dict[str, Any] = {
            "noise": initial_noise,
            "realtime_diagnostics": diagnostics,
        }
        action_origin_step = request_step

        if self.realtime_method == "rtc":
            kwargs.update(
                {
                    "inference_delay": 0 if warm_start else self.fixed_delay_steps,
                    "prev_chunk_left_over": previous_chunk,
                    "prefix_attention_horizon": self.rtc_prefix_attention_horizon,
                }
            )
        elif self.realtime_method == "m1r1" and not warm_start:
            refresh_offset = (self.fixed_delay_steps + 1) // 2
            target_step = request_step + refresh_offset
            kwargs.update(
                {
                    "refresh_batch_provider": lambda: self._wait_for_refresh_batch(
                        target_step=target_step,
                        episode_generation=episode_generation,
                    ),
                    "refresh_after_step": self.refresh_after_step,
                }
            )
            action_origin_step = target_step
        elif self.realtime_method == "m3_fve" and not warm_start:
            action_origin_step = request_step + m3_fve_forward_steps

        amp_context = (
            self._torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.use_amp
            else contextlib.nullcontext()
        )
        with amp_context:
            actions = self.policy.predict_action_chunk(batch, **kwargs)
            original = actions.squeeze(0).detach().clone()
            processed = self.postprocessor(actions).squeeze(0)
        if hasattr(processed, "detach"):
            processed = processed.detach().cpu().numpy()
        processed = np.asarray(processed, dtype=np.float32)
        if processed.ndim != 2:
            raise ValueError(f"Policy returned chunk shape {processed.shape}, expected [H,A]")
        if processed.shape[0] != self.prediction_horizon:
            raise ValueError(
                f"Policy returned horizon {processed.shape[0]}, expected {self.prediction_horizon}"
            )
        if processed.shape[1] < self.action_dim:
            raise ValueError(
                f"Policy returned {processed.shape[1]} action dims, expected {self.action_dim}"
            )
        return {
            "request_step": request_step,
            "action_origin_step": action_origin_step,
            "warm_start": warm_start,
            "original": original,
            "processed": processed[:, : self.action_dim].copy(),
            "latency_seconds": time.perf_counter() - started,
            "diagnostics": diagnostics,
            "initial_noise": initial_noise.detach().clone(),
            "noise_origin_step": request_step,
            "coupling_origin_shift_steps": coupling_origin_shift,
            "request_state": request_state.copy(),
            "predicted_old_states": (
                None if predicted_old_states is None else predicted_old_states.copy()
            ),
            "m3_fve_state": (
                None if m3_fve_state is None else m3_fve_state.copy()
            ),
            "m3_fve_forward_steps": m3_fve_forward_steps,
        }

    def _install_completion(self, completion: dict[str, Any], completion_step: int) -> None:
        crop = max(0, completion_step - int(completion["action_origin_step"]))
        original = completion["original"]
        processed = completion["processed"].copy()
        if crop >= int(processed.shape[0]):
            raise RuntimeError(
                f"Completed chunk is entirely stale: crop={crop}, horizon={processed.shape[0]}"
            )
        sit_correction_l2 = None
        sit_raw_saturated_dimensions = None
        sit_request_arm_rmse = None
        if (
            self.realtime_method == "m3_sit"
            and not completion["warm_start"]
            and completion["predicted_old_states"] is not None
        ):
            assert self._action_lower is not None and self._action_upper is not None
            predicted_old_states = completion["predicted_old_states"]
            processed, correction, sit_raw_saturated_dimensions = (
                _apply_absolute_qpos_sit(
                    processed,
                    crop_steps=crop,
                    predicted_old_states=predicted_old_states,
                    measured_request_state=completion["request_state"],
                    action_lower=self._action_lower,
                    action_upper=self._action_upper,
                )
            )
            sit_correction_l2 = float(np.linalg.norm(correction))
            sit_request_arm_rmse = float(np.sqrt(np.mean(np.square(correction))))

        self._original_queue = deque(original[crop:].unbind(0))
        self._processed_queue = deque(processed[crop:])
        actual_handoff_state = np.asarray(
            self._observation_history[completion_step]["observation"][
                "observation.state"
            ],
            dtype=np.float32,
        ).reshape(-1)
        self._predicted_qpos_states = deque(
            _absolute_qpos_predicted_states(
                actual_handoff_state,
                processed[crop:, : self.action_dim],
            )
        )
        self._previous_initial_noise = completion["initial_noise"].detach().clone()
        self._previous_noise_origin_step = int(completion["noise_origin_step"])

        fve_state_rmse = None
        fve_arm_rmse = None
        fve_gripper_abs_mean = None
        if completion["m3_fve_state"] is not None:
            fve_error = completion["m3_fve_state"] - actual_handoff_state
            arm_indices = [
                index
                for index in range(self.action_dim)
                if index not in ROBOTWIN_DUAL_ALOHA_GRIPPER_INDICES
            ]
            fve_state_rmse = float(np.sqrt(np.mean(np.square(fve_error))))
            fve_arm_rmse = float(
                np.sqrt(np.mean(np.square(fve_error[arm_indices])))
            )
            fve_gripper_abs_mean = float(
                np.mean(
                    np.abs(
                        fve_error[list(ROBOTWIN_DUAL_ALOHA_GRIPPER_INDICES)]
                    )
                )
            )
        self._emit_event(
            {
                "event": "completion",
                "request_step": int(completion["request_step"]),
                "completion_step": completion_step,
                "actual_delay_steps": completion_step - int(completion["request_step"]),
                "action_origin_step": int(completion["action_origin_step"]),
                "crop_steps": crop,
                "latency_seconds": float(completion["latency_seconds"]),
                "warm_start": bool(completion["warm_start"]),
                "queue_length_after_install": len(self._processed_queue),
                "diagnostics": completion["diagnostics"],
                "c2_rho": self.c2_rho if self.realtime_method == "c2" else None,
                "coupling_origin_shift_steps": completion[
                    "coupling_origin_shift_steps"
                ],
                "m3_fve_forward_steps": completion["m3_fve_forward_steps"],
                "m3_fve_state_rmse_to_actual_handoff": fve_state_rmse,
                "m3_fve_arm_rmse_to_actual_handoff": fve_arm_rmse,
                "m3_fve_gripper_abs_mean_to_actual_handoff": (
                    fve_gripper_abs_mean
                ),
                "m3_sit_correction_l2": sit_correction_l2,
                "m3_sit_request_arm_rmse": sit_request_arm_rmse,
                "m3_sit_raw_saturated_dimensions": (
                    sit_raw_saturated_dimensions
                ),
                "training_updates": 0,
            }
        )

    def _launch_request(self, payload: dict[str, Any], request_step: int) -> None:
        previous_chunk = (
            self._torch.stack(list(self._original_queue)).unsqueeze(0)
            if self._original_queue
            else None
        )
        previous_processed = (
            np.stack(list(self._processed_queue)).astype(np.float32, copy=True)
            if self._processed_queue
            else np.empty((0, self.action_dim), dtype=np.float32)
        )
        predicted_old_states = (
            np.stack(list(self._predicted_qpos_states)).astype(
                np.float32, copy=True
            )
            if self._predicted_qpos_states
            else None
        )
        episode_generation = self._episode_generation
        request_payload = copy.deepcopy(payload)

        def run() -> None:
            try:
                completion = self._predict_chunk(
                    request_payload,
                    request_step=request_step,
                    warm_start=False,
                    previous_chunk=previous_chunk,
                    previous_processed=previous_processed,
                    predicted_old_states=predicted_old_states,
                    episode_generation=episode_generation,
                )
                with self._condition:
                    if (
                        not self._cancelled
                        and episode_generation == self._episode_generation
                    ):
                        self._completed = completion
                    self._condition.notify_all()
            except BaseException as exc:
                with self._condition:
                    self._worker_error = exc
                    self._condition.notify_all()

        self._worker = threading.Thread(
            target=run,
            daemon=True,
            name=f"robotwin-{self.realtime_method}-chunk",
        )
        self._worker.start()
        self._emit_event(
            {
                "event": "request",
                "request_step": request_step,
                "previous_chunk_length": (
                    0 if previous_chunk is None else int(previous_chunk.shape[-2])
                ),
                "previous_processed_length": len(previous_processed),
            }
        )

    def _install_due_background_completion(self) -> None:
        with self._condition:
            worker = self._worker
            due = (
                worker is not None
                and self._control_step
                >= self._next_request_step - self.execution_horizon + self.fixed_delay_steps
            )
        if not due:
            return
        worker.join()
        with self._condition:
            if self._worker_error is not None:
                error = self._worker_error
                self._worker_error = None
                self._worker = None
                raise RuntimeError("RoboTwin realtime policy worker failed") from error
            completion = self._completed
            if completion is None:
                raise RuntimeError("Realtime worker exited without a completed chunk")
            self._install_completion(completion, self._control_step)
            self._completed = None
            self._worker = None

    def act(self, payload: dict[str, Any]) -> np.ndarray:
        with self._condition:
            step = self._control_step
            self._observation_history[step] = copy.deepcopy(payload)
            oldest = step - self.prediction_horizon
            for history_step in tuple(self._observation_history):
                if history_step < oldest:
                    del self._observation_history[history_step]
            self._condition.notify_all()

        if not self._initialized:
            completion = self._predict_chunk(
                copy.deepcopy(payload),
                request_step=step,
                warm_start=True,
                previous_chunk=None,
                previous_processed=np.empty(
                    (0, self.action_dim), dtype=np.float32
                ),
                predicted_old_states=None,
                episode_generation=self._episode_generation,
            )
            with self._condition:
                self._install_completion(completion, step)
                self._initialized = True
                self._next_request_step = (
                    self.execution_horizon
                    if self.realtime_method == "sync"
                    else 0
                )

        if self.realtime_method == "sync":
            if step >= self._next_request_step:
                completion = self._predict_chunk(
                    copy.deepcopy(payload),
                    request_step=step,
                    warm_start=False,
                    previous_chunk=None,
                    previous_processed=np.stack(
                        list(self._processed_queue)
                    ).astype(np.float32, copy=True),
                    predicted_old_states=np.stack(
                        list(self._predicted_qpos_states)
                    ).astype(np.float32, copy=True),
                    episode_generation=self._episode_generation,
                )
                with self._condition:
                    self._install_completion(completion, step)
                    self._next_request_step = step + self.execution_horizon
        else:
            self._install_due_background_completion()
            with self._condition:
                if self._worker is None and step >= self._next_request_step:
                    self._launch_request(payload, step)
                    self._next_request_step = step + self.execution_horizon
            self._install_due_background_completion()

        with self._condition:
            if not self._processed_queue or not self._original_queue:
                raise RuntimeError("Action queue underrun")
            action = np.asarray(self._processed_queue.popleft(), dtype=np.float32).reshape(-1)
            self._original_queue.popleft()
            if not self._predicted_qpos_states:
                raise RuntimeError("Predicted qpos state queue underrun")
            self._predicted_qpos_states.popleft()
            self._control_step += 1

        return action


def _encode_obs(observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    raw_obs = observation["observation"]
    joint_action = observation["joint_action"]
    return {
        "observation": {
            "observation.images.head_camera": np.asarray(raw_obs["head_camera"]["rgb"], dtype=np.uint8),
            "observation.images.left_camera": np.asarray(raw_obs["left_camera"]["rgb"], dtype=np.uint8),
            "observation.images.right_camera": np.asarray(raw_obs["right_camera"]["rgb"], dtype=np.uint8),
            "observation.state": np.asarray(joint_action["vector"], dtype=np.float32),
        },
        "instruction": instruction,
    }


def get_model(usr_args: dict[str, Any]) -> LeRobotPolicyAdapter:
    return LeRobotPolicyAdapter(usr_args)


def eval(TASK_ENV, model, observation):
    """RoboTwin policy hook.

    In upstream eval_policy_client.py, `model` is a ModelClient. In direct
    eval_policy.py, it can be a LeRobotPolicyAdapter. Both expose the same
    logical `act` method.
    """

    instruction = TASK_ENV.get_instruction()
    _debug(f"encode_obs start step={getattr(TASK_ENV, 'take_action_cnt', 'n/a')}")
    payload = _encode_obs(observation, instruction)
    _debug(f"encode_obs done step={getattr(TASK_ENV, 'take_action_cnt', 'n/a')}")

    if hasattr(model, "call"):
        _debug(f"model.call start step={getattr(TASK_ENV, 'take_action_cnt', 'n/a')}")
        action = model.call(func_name="act", obs=payload)
        _debug(f"model.call done step={getattr(TASK_ENV, 'take_action_cnt', 'n/a')}")
    else:
        _debug(f"model.act start step={getattr(TASK_ENV, 'take_action_cnt', 'n/a')}")
        action = model.act(payload)
        _debug(f"model.act done step={getattr(TASK_ENV, 'take_action_cnt', 'n/a')}")

    _debug(f"take_action start step={getattr(TASK_ENV, 'take_action_cnt', 'n/a')}")
    TASK_ENV.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")
    _debug(f"take_action done step={getattr(TASK_ENV, 'take_action_cnt', 'n/a')}")


def reset_model(model):
    if hasattr(model, "call"):
        return model.call(func_name="reset_model")
    return model.reset_model()
