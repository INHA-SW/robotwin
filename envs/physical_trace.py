"""Append-only physical trajectory diagnostics for RoboTwin rollouts.

The recorder is inert unless ``ROBOTWIN_PHYSICAL_TRACE_RECORDS`` is set.  It
captures policy-waypoint boundaries, realized robot/object state, TOPP output,
and the first appearance of task-relevant contact pairs.  It intentionally
does not alter control targets or simulator stepping.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any) -> list:
    return np.asarray(value, dtype=np.float64).tolist()


def _pose(value: Any) -> list[float]:
    return _array(np.concatenate((np.asarray(value.p), np.asarray(value.q))))


def _optional_array(value: Any, name: str) -> list | None:
    attribute = getattr(value, name, None)
    if attribute is None:
        return None
    try:
        result = attribute() if callable(attribute) else attribute
        return _array(result)
    except Exception:
        return None


class PhysicalTraceRecorder:
    """Record task-relevant physical state without changing rollout behavior."""

    SCHEMA_VERSION = 1
    POINT_TYPES = ("contact", "target", "functional", "orientation")

    @classmethod
    def from_environment(cls, environment: Any) -> "PhysicalTraceRecorder | None":
        output = os.environ.get("ROBOTWIN_PHYSICAL_TRACE_RECORDS", "").strip()
        if not output:
            return None
        mode = os.environ.get("ROBOTWIN_PHYSICAL_TRACE_MODE", "full").strip()
        if mode not in {"full", "handover_margin"}:
            raise ValueError(
                "ROBOTWIN_PHYSICAL_TRACE_MODE must be full or handover_margin, "
                f"got {mode!r}"
            )
        return cls(environment, Path(output), mode=mode)

    def __init__(self, environment: Any, output: Path, *, mode: str = "full"):
        self.environment = environment
        self.output = output
        self.mode = mode
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.physics_step_index = 0
        self.waypoint_control_step = 0
        self.seen_contact_pairs: set[tuple[str, str]] = set()
        self.current_context: dict[str, Any] = {}
        self.task_objects = self._discover_task_objects()
        self.task_entity_names = self._discover_task_entity_names()
        if self.mode == "handover_margin":
            if str(self.environment.task_name) != "handover_mic":
                raise ValueError(
                    "handover_margin trace is only valid for handover_mic, got "
                    f"{self.environment.task_name!r}"
                )
            initial_state = self._handover_margin_snapshot()
        else:
            initial_state = self._state_snapshot()
            self.seen_contact_pairs.update(
                tuple(contact["pair_key"])
                for contact in initial_state["contacts"]
            )
        self._append(
            "episode_start",
            {
                "trace_mode": self.mode,
                "task": str(self.environment.task_name),
                "episode_index": int(self.environment.ep_num),
                "task_objects": sorted(self.task_objects),
                "task_entity_names": sorted(self.task_entity_names),
                "state": initial_state,
            },
        )

    def _append(self, record_kind: str, payload: dict[str, Any]) -> None:
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "record_kind": record_kind,
            "wall_time_unix": time.time(),
            "task": str(self.environment.task_name),
            "episode_index": int(self.environment.ep_num),
            **self.current_context,
            **payload,
        }
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()

    def _discover_task_objects(self) -> dict[str, Any]:
        result = {}
        for attribute_name, value in vars(self.environment).items():
            if not hasattr(value, "actor") or not hasattr(value, "get_pose"):
                continue
            if not hasattr(value, "config") or not hasattr(value, "POINTS"):
                continue
            result[str(attribute_name)] = value
        return result

    def _discover_task_entity_names(self) -> set[str]:
        names: set[str] = set()
        for value in self.task_objects.values():
            try:
                names.add(str(value.get_name()))
            except Exception:
                pass
            actor = getattr(value, "actor", None)
            get_links = getattr(actor, "get_links", None)
            if callable(get_links):
                try:
                    names.update(str(link.get_name()) for link in get_links())
                except Exception:
                    pass
        names.discard("")
        return names

    def _semantic_points(self, value: Any) -> dict[str, list[dict[str, Any]]]:
        points: dict[str, list[dict[str, Any]]] = {}
        mapping = getattr(value, "POINTS", {})
        config = getattr(value, "config", {})
        for point_type in self.POINT_TYPES:
            config_key = mapping.get(point_type)
            entries = config.get(config_key, []) if config_key is not None else []
            if entries is None:
                continue
            if not isinstance(entries, (list, tuple)):
                entries = [entries]
            values = []
            for index in range(len(entries)):
                try:
                    point = value.get_point(point_type, index, "pose")
                except Exception:
                    continue
                if point is not None:
                    values.append({"index": index, "pose_pq_wxyz": _pose(point)})
            if values:
                points[point_type] = values
        return points

    def _object_states(self) -> dict[str, dict[str, Any]]:
        states = {}
        for attribute_name, value in self.task_objects.items():
            try:
                pose = _pose(value.get_pose())
                entity_name = str(value.get_name())
            except Exception:
                continue
            state: dict[str, Any] = {
                "entity_name": entity_name,
                "pose_pq_wxyz": pose,
                "semantic_points": self._semantic_points(value),
            }
            qpos = _optional_array(value, "get_qpos")
            qvel = _optional_array(value, "get_qvel")
            if qpos is not None:
                state["qpos"] = qpos
            if qvel is not None:
                state["qvel"] = qvel
            actor = getattr(value, "actor", None)
            linear_velocity = _optional_array(actor, "get_linear_velocity")
            angular_velocity = _optional_array(actor, "get_angular_velocity")
            if linear_velocity is not None:
                state["linear_velocity"] = linear_velocity
            if angular_velocity is not None:
                state["angular_velocity"] = angular_velocity
            states[attribute_name] = state
        return states

    @staticmethod
    def _contact_point(point: Any) -> dict[str, Any]:
        result = {}
        for name in ("position", "normal", "impulse"):
            if hasattr(point, name):
                result[name] = _array(getattr(point, name))
        if hasattr(point, "separation"):
            result["separation"] = float(point.separation)
        return result

    def _relevant_contacts(self) -> list[dict[str, Any]]:
        contacts = []
        for contact in self.environment.scene.get_contacts():
            first = str(contact.bodies[0].entity.name)
            second = str(contact.bodies[1].entity.name)
            if not (
                first in self.task_entity_names
                or second in self.task_entity_names
            ):
                continue
            contacts.append(
                {
                    "pair": [first, second],
                    "pair_key": sorted((first, second)),
                    "points": [self._contact_point(point) for point in contact.points],
                }
            )
        return contacts

    def _contact_summary(self) -> dict[str, Any]:
        contacts = self._relevant_contacts()
        impulse_norms = []
        separations = []
        point_count = 0
        for contact in contacts:
            for point in contact["points"]:
                point_count += 1
                if "impulse" in point:
                    impulse_norms.append(float(np.linalg.norm(point["impulse"])))
                if "separation" in point:
                    separations.append(float(point["separation"]))
        return {
            "pair_count": len(contacts),
            "point_count": point_count,
            "pairs": [contact["pair"] for contact in contacts],
            "impulse_norm_sum": float(sum(impulse_norms)),
            "impulse_norm_max": float(max(impulse_norms, default=0.0)),
            "separation_min": float(min(separations)) if separations else None,
        }

    def _handover_margin_snapshot(
        self, *, include_contact_summary: bool = False
    ) -> dict[str, Any]:
        environment = self.environment
        microphone_point = np.asarray(
            environment.microphone.get_functional_point(0), dtype=np.float64
        ).reshape(-1)
        if microphone_point.shape[0] < 3:
            raise ValueError(
                "handover microphone functional point must contain xyz, got "
                f"{microphone_point.shape}"
            )
        receiver_is_left = environment.handover_arm_tag == "left"
        receiver_arm = "left" if receiver_is_left else "right"
        giver_arm = "right" if receiver_is_left else "left"
        receiver_closed = bool(
            environment.is_left_gripper_close()
            if receiver_is_left
            else environment.is_right_gripper_close()
        )
        giver_open = bool(
            environment.is_right_gripper_open()
            if receiver_is_left
            else environment.is_left_gripper_open()
        )
        contact_positions = environment.get_gripper_actor_contact_position(
            "018_microphone"
        )
        contact_present = len(contact_positions) > 0
        height_margin = float(microphone_point[2] - 0.92)
        receiver_side_margin = float(
            -microphone_point[0] if receiver_is_left else microphone_point[0]
        )
        height_ok = height_margin > 0.0
        receiver_side_ok = receiver_side_margin > 0.0
        left_tcp = np.asarray(environment.robot.get_left_tcp_pose(), dtype=np.float64)
        right_tcp = np.asarray(environment.robot.get_right_tcp_pose(), dtype=np.float64)
        receiver_tcp = left_tcp if receiver_is_left else right_tcp
        giver_tcp = right_tcp if receiver_is_left else left_tcp
        predicate_success = bool(
            receiver_closed
            and giver_open
            and contact_present
            and height_ok
            and receiver_side_ok
        )
        state = {
            "environment_step": int(environment.take_action_cnt),
            "physics_step_index": int(self.physics_step_index),
            "eval_success": bool(environment.eval_success),
            "receiver_arm": receiver_arm,
            "giver_arm": giver_arm,
            "microphone_pose_pq_wxyz": _pose(environment.microphone.get_pose()),
            "microphone_functional_xyz": _array(microphone_point[:3]),
            "receiver_closed": receiver_closed,
            "giver_open": giver_open,
            "contact_present": bool(contact_present),
            "height_margin_z_minus_0p92": height_margin,
            "receiver_side_signed_x_margin": receiver_side_margin,
            "height_ok": bool(height_ok),
            "receiver_side_ok": bool(receiver_side_ok),
            "predicate_success": predicate_success,
            "left_gripper_value": _array(environment.robot.get_left_gripper_val()),
            "right_gripper_value": _array(environment.robot.get_right_gripper_val()),
            "left_tcp_pq_wxyz": _array(left_tcp),
            "right_tcp_pq_wxyz": _array(right_tcp),
            "receiver_tcp_to_microphone_distance": float(
                np.linalg.norm(receiver_tcp[:3] - microphone_point[:3])
            ),
            "giver_tcp_to_microphone_distance": float(
                np.linalg.norm(giver_tcp[:3] - microphone_point[:3])
            ),
            "left_real_arm_state": _array(
                environment.robot.get_left_arm_real_jointState()
            ),
            "right_real_arm_state": _array(
                environment.robot.get_right_arm_real_jointState()
            ),
        }
        if include_contact_summary:
            state["contact_summary"] = self._contact_summary()
        return state

    def _update_handover_margin_aggregate(
        self, state: dict[str, Any], *, control_step: int
    ) -> None:
        predicates = {
            "receiver_closed": bool(state["receiver_closed"]),
            "giver_open": bool(state["giver_open"]),
            "contact_present": bool(state["contact_present"]),
            "height_ok": bool(state["height_ok"]),
            "receiver_side_ok": bool(state["receiver_side_ok"]),
        }
        for name, value in predicates.items():
            if value:
                self._handover_ever_true[name] = True
                if self._handover_first_true_control_step[name] is None:
                    self._handover_first_true_control_step[name] = int(control_step)
        true_count = sum(predicates.values())
        if (
            self._handover_closest_state is None
            or true_count > self._handover_closest_true_count
        ):
            self._handover_closest_true_count = int(true_count)
            self._handover_closest_state = {
                "control_step": int(control_step),
                "true_predicate_count": int(true_count),
                "missing_predicates": [
                    name for name, value in predicates.items() if not value
                ],
                "state": state,
            }

    def _state_snapshot(self) -> dict[str, Any]:
        robot = self.environment.robot
        articulation = robot.left_entity
        return {
            "environment_step": int(self.environment.take_action_cnt),
            "physics_step_index": int(self.physics_step_index),
            "eval_success": bool(self.environment.eval_success),
            "robot_articulation_qpos": _array(articulation.get_qpos()),
            "robot_articulation_qvel": _array(articulation.get_qvel()),
            "left_real_arm_state": _array(robot.get_left_arm_real_jointState()),
            "right_real_arm_state": _array(robot.get_right_arm_real_jointState()),
            "left_tcp_pq_wxyz": _array(robot.get_left_tcp_pose()),
            "right_tcp_pq_wxyz": _array(robot.get_right_tcp_pose()),
            "objects": self._object_states(),
            "contacts": self._relevant_contacts(),
        }

    def start_waypoint(
        self,
        action: np.ndarray,
        *,
        action_type: str,
        replan_index: int,
        chunk_action_index: int,
    ) -> None:
        self.current_context = {
            "replan_index": int(replan_index),
            "chunk_action_index": int(chunk_action_index),
            "environment_step": int(self.environment.take_action_cnt),
        }
        self.waypoint_control_step = 0
        if self.mode == "handover_margin":
            predicate_names = (
                "receiver_closed",
                "giver_open",
                "contact_present",
                "height_ok",
                "receiver_side_ok",
            )
            self._handover_target_action = _array(action)
            self._handover_action_type = str(action_type)
            self._handover_ever_true = {name: False for name in predicate_names}
            self._handover_first_true_control_step = {
                name: None for name in predicate_names
            }
            self._handover_closest_state = None
            self._handover_closest_true_count = -1
            self._handover_start_state = self._handover_margin_snapshot()
            self._update_handover_margin_aggregate(
                self._handover_start_state, control_step=0
            )
            return
        self._append(
            "waypoint_start",
            {
                "action_type": str(action_type),
                "target_action": _array(action),
                "state": self._state_snapshot(),
            },
        )

    def record_topp_plan(
        self,
        arm: str,
        *,
        input_path: np.ndarray,
        succeeded: bool,
        positions: np.ndarray | None,
        velocities: np.ndarray | None,
        planner_duration: float | None,
        exception: str | None = None,
    ) -> None:
        if self.mode == "handover_margin":
            return
        self._append(
            "topp_plan",
            {
                "arm": str(arm),
                "succeeded": bool(succeeded),
                "input_path": _array(input_path),
                "positions": _array(positions) if positions is not None else None,
                "velocities": _array(velocities) if velocities is not None else None,
                "planner_duration": (
                    float(planner_duration) if planner_duration is not None else None
                ),
                "exception": exception,
            },
        )

    def observe_control_step(self) -> None:
        self.physics_step_index += 1
        self.waypoint_control_step += 1
        if self.mode == "handover_margin":
            state = self._handover_margin_snapshot()
            self._update_handover_margin_aggregate(
                state, control_step=self.waypoint_control_step
            )
            return
        contacts = self._relevant_contacts()
        new_pairs = []
        for contact in contacts:
            pair = tuple(contact["pair_key"])
            if pair not in self.seen_contact_pairs:
                self.seen_contact_pairs.add(pair)
                new_pairs.append(contact)
        if new_pairs:
            self._append(
                "contact_onset",
                {
                    "waypoint_control_step": int(self.waypoint_control_step),
                    "new_contacts": new_pairs,
                    "state": self._state_snapshot(),
                },
            )

    def end_waypoint(self, *, success: bool) -> None:
        if self.mode == "handover_margin":
            endpoint_state = self._handover_margin_snapshot(
                include_contact_summary=True
            )
            self._update_handover_margin_aggregate(
                endpoint_state, control_step=self.waypoint_control_step
            )
            self._append(
                "waypoint_end",
                {
                    "trace_mode": self.mode,
                    "action_type": self._handover_action_type,
                    "target_action": self._handover_target_action,
                    "success_after_waypoint": bool(success),
                    "waypoint_control_steps": int(self.waypoint_control_step),
                    "start_state": self._handover_start_state,
                    "endpoint_state": endpoint_state,
                    "ever_true": self._handover_ever_true,
                    "first_true_control_step": self._handover_first_true_control_step,
                    "closest_state": self._handover_closest_state,
                },
            )
            return
        self._append(
            "waypoint_end",
            {
                "success_after_waypoint": bool(success),
                "waypoint_control_steps": int(self.waypoint_control_step),
                "state": self._state_snapshot(),
            },
        )
