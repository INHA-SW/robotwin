import sys
import os
import subprocess
import fcntl
import json
import faulthandler

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _append_episode_record(record):
    output_path = os.environ.get("ROBOTWIN_EPISODE_RECORDS")
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def _write_initial_probe_artifact(
    probe_root,
    *,
    task_name,
    task_config,
    seed,
    episode_index,
    instruction,
    noise_seed,
    features,
):
    root = Path(probe_root)
    output_path = root / task_config / task_name / f"seed_{seed}_episode_{episode_index}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Initial solver probe already exists: {output_path}")

    trace_budgets = np.asarray(
        features.get("trace_budgets", np.asarray([], dtype=np.int64)),
        dtype=np.int64,
    )
    ensemble_noise_seeds = np.asarray(
        features.get("ensemble_noise_seeds", np.asarray([], dtype=np.int64)),
        dtype=np.int64,
    )
    conditional_residual = "conditional_residual_sq" in features
    schema_version = (
        6
        if conditional_residual
        else (5 if ensemble_noise_seeds.size else (4 if trace_budgets.size else 3))
    )
    noise_contract = (
        "fixed_solver_noise_plus_independent_paired_conditional_residual_noises"
        if conditional_residual
        else (
            "consecutive_manual_seed_single_sample_traces_at_one_observation"
            if ensemble_noise_seeds.size
            else "first_torch_normal_after_scene_seed_reset"
        )
    )
    payload = {
        "schema_version": np.asarray(schema_version, dtype=np.int64),
        "task": np.asarray(str(task_name)),
        "task_config": np.asarray(str(task_config)),
        "seed": np.asarray(int(seed), dtype=np.int64),
        "episode_index": np.asarray(int(episode_index), dtype=np.int64),
        "instruction": np.asarray(str(instruction)),
        "noise_seed": np.asarray(int(noise_seed), dtype=np.int64),
        "noise_contract": np.asarray(noise_contract),
    }
    for key, value in features.items():
        payload[str(key)] = np.asarray(value)

    temporary_path = output_path.with_suffix(f".{os.getpid()}.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary_path, output_path)
    return output_path


def _load_episode_manifest(task_name, task_config, expected_count):
    """Load exact seed/instruction rows from a prior accepted rollout ledger."""

    manifest_text = os.environ.get("ROBOTWIN_EPISODE_MANIFEST", "").strip()
    if not manifest_text:
        return None
    path = Path(manifest_text)
    if not path.is_file():
        raise FileNotFoundError(f"Missing RoboTwin episode manifest: {path}")

    selected = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        if (
            str(record.get("task")) == str(task_name)
            and str(record.get("task_config")) == str(task_config)
        ):
            if record.get("included_in_denominator") is not True:
                raise ValueError(
                    f"{path}:{line_number}: manifest row was not in its source denominator"
                )
            if record.get("simulator_error") is not None:
                raise ValueError(
                    f"{path}:{line_number}: manifest row has a simulator error"
                )
            if "seed" not in record or not str(record.get("instruction", "")).strip():
                raise ValueError(
                    f"{path}:{line_number}: manifest row needs seed and instruction"
                )
            selected.append(
                {
                    "seed": int(record["seed"]),
                    "instruction": str(record["instruction"]),
                    "source_run_id": record.get("run_id"),
                    "source_episode_uid": record.get("episode_uid"),
                }
            )

    if len(selected) < expected_count:
        raise ValueError(
            f"{path}: needs at least {expected_count} rows for {task_config}/{task_name}, "
            f"found {len(selected)}"
        )
    selected = selected[:expected_count]
    seeds = [row["seed"] for row in selected]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: duplicate reset seed for {task_config}/{task_name}")
    print(
        f"[arm] fixed episode manifest: path={path} task={task_name} "
        f"rows={len(selected)} seeds={seeds}",
        flush=True,
    )
    return selected


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(os.environ.get("ROBOTWIN_TEST_NUM", "100"))
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")
    initial_probe_root = os.environ.get("ROBOTWIN_INITIAL_SOLVER_PROBE_DIR", "").strip()
    probe_func = (
        eval_function_decorator(policy_name, "probe")
        if initial_probe_root
        else None
    )

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    episode_manifest = _load_episode_manifest(
        task_name,
        args["task_config"],
        test_num,
    )
    manifest_entry = None

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if episode_manifest is not None:
            manifest_entry = episode_manifest[now_id]
            now_seed = int(manifest_entry["seed"])
            instruction = str(manifest_entry["instruction"])
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        elif expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                print("Unexpected error during expert seed validation:", repr(e))
                traceback.print_exc()
                raise

        if episode_manifest is None:
            if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
                succ_seed += 1
                suc_test_seed_list.append(now_seed)
            else:
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        if episode_manifest is None:
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if initial_probe_root:
            reset_func(model)
            observation = TASK_ENV.get_obs()
            noise_seed = _env_int("ROBOTWIN_PROBE_NOISE_SEED_BASE", 0) + int(now_seed)
            features = probe_func(
                TASK_ENV,
                model,
                observation,
                noise_seed=noise_seed,
            )
            artifact_path = _write_initial_probe_artifact(
                initial_probe_root,
                task_name=task_name,
                task_config=args["task_config"],
                seed=now_seed,
                episode_index=now_id,
                instruction=instruction,
                noise_seed=noise_seed,
                features=features,
            )
            trace_budgets = np.asarray(
                features.get("trace_budgets", np.asarray([], dtype=np.int64)),
                dtype=np.int64,
            ).tolist()
            ensemble_noise_seeds = np.asarray(
                features.get("ensemble_noise_seeds", np.asarray([], dtype=np.int64)),
                dtype=np.int64,
            ).tolist()
            conditional_residual = "conditional_residual_sq" in features
            schema_version = (
                6
                if conditional_residual
                else (5 if ensemble_noise_seeds else (4 if trace_budgets else 3))
            )
            noise_contract = (
                "fixed_solver_noise_plus_independent_paired_conditional_residual_noises"
                if conditional_residual
                else (
                    "consecutive_manual_seed_single_sample_traces_at_one_observation"
                    if ensemble_noise_seeds
                    else "first_torch_normal_after_scene_seed_reset"
                )
            )
            _append_episode_record(
                {
                    "schema_version": schema_version,
                    "noise_contract": noise_contract,
                    "record_kind": (
                        "conditional_candidate_consistency_probe"
                        if conditional_residual
                        else (
                            "full_solver_trace_ensemble_probe"
                            if ensemble_noise_seeds
                            else (
                                "full_solver_trace_probe"
                                if trace_budgets
                                else "initial_solver_probe"
                            )
                        )
                    ),
                    "created_at": datetime.now().astimezone().isoformat(),
                    "run_id": os.environ.get("ARM_EVAL_RUN_ID"),
                    "model": os.environ.get("ARM_EVAL_MODEL", policy_name),
                    "variant": os.environ.get("ARM_EVAL_VARIANT"),
                    "benchmark": "robotwin",
                    "task": task_name,
                    "task_config": args["task_config"],
                    "seed": int(now_seed),
                    "episode_index": int(now_id),
                    "instruction": str(instruction),
                    "noise_seed": int(noise_seed),
                    "ensemble_noise_seeds": ensemble_noise_seeds,
                    "solver_trace_budgets": trace_budgets,
                    "action_executed": False,
                    "included_in_denominator": False,
                    "termination_reason": "no_motion_probe",
                    "artifact_path": str(artifact_path),
                    "feature_shapes": {
                        key: list(np.asarray(value).shape)
                        for key, value in features.items()
                    },
                    "manifest_path": os.environ.get("ROBOTWIN_EPISODE_MANIFEST"),
                    "manifest_sha256": os.environ.get(
                        "ROBOTWIN_EPISODE_MANIFEST_SHA256"
                    ),
                    "source_run_id": (
                        manifest_entry.get("source_run_id")
                        if manifest_entry is not None
                        else None
                    ),
                    "source_episode_uid": (
                        manifest_entry.get("source_episode_uid")
                        if manifest_entry is not None
                        else None
                    ),
                }
            )
            now_id += 1
            TASK_ENV.test_num += 1
            TASK_ENV.close_env(
                clear_cache=((succ_seed + 1) % clear_cache_freq == 0)
            )
            print(
                f"[arm] no-motion initial solver probe saved: {artifact_path}",
                flush=True,
            )
            now_seed += 1
            continue

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        episode_record = {
            "schema_version": 2,
            "record_kind": "policy_rollout",
            "created_at": datetime.now().astimezone().isoformat(),
            "run_id": os.environ.get("ARM_EVAL_RUN_ID"),
            "model": os.environ.get("ARM_EVAL_MODEL", policy_name),
            "variant": os.environ.get("ARM_EVAL_VARIANT"),
            "integration_schedule": (
                "terminal_jump"
                if os.environ.get("PI05_TERMINAL_JUMP_TIME", "").strip()
                else "uniform_euler"
            ),
            "terminal_jump_time": (
                float(os.environ["PI05_TERMINAL_JUMP_TIME"])
                if os.environ.get("PI05_TERMINAL_JUMP_TIME", "").strip()
                else None
            ),
            "action_intervention_mode": (
                os.environ.get("PI05_ACTION_INTERVENTION_MODE", "").strip()
                or None
            ),
            "intervention_horizons_1based": (
                os.environ.get("PI05_INTERVENTION_HORIZONS_1BASED", "").strip()
                or None
            ),
            "intervention_actions_1based": (
                os.environ.get("PI05_INTERVENTION_ACTIONS_1BASED", "").strip()
                or None
            ),
            "benchmark": "robotwin",
            "checkpoint_step": _env_int("ARM_EVAL_STEP", 0),
            "task": task_name,
            "task_config": args["task_config"],
            "seed": int(now_seed),
            "episode_index": int(now_id),
            "success": bool(succ),
            "episode_steps": int(TASK_ENV.take_action_cnt),
            "step_limit": int(TASK_ENV.step_lim),
            "termination_reason": "success" if succ else "step_limit",
            "instruction": str(instruction),
            "episode_source": (
                "fixed_manifest" if episode_manifest is not None else "expert_filter"
            ),
            "manifest_path": (
                os.environ.get("ROBOTWIN_EPISODE_MANIFEST")
                if episode_manifest is not None
                else None
            ),
            "manifest_sha256": (
                os.environ.get("ROBOTWIN_EPISODE_MANIFEST_SHA256")
                if episode_manifest is not None
                else None
            ),
            "source_run_id": (
                manifest_entry.get("source_run_id")
                if episode_manifest is not None
                else None
            ),
            "source_episode_uid": (
                manifest_entry.get("source_episode_uid")
                if episode_manifest is not None
                else None
            ),
            "included_in_denominator": True,
            "simulator_error": None,
        }
        episode_record["episode_uid"] = ":".join(
            str(episode_record[key])
            for key in ("run_id", "checkpoint_step", "task_config", "task", "seed")
        )
        _append_episode_record(episode_record)

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    hang_trace_seconds = _env_int("ROBOTWIN_HANG_TRACE_SECONDS", 0)
    if hang_trace_seconds > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(hang_trace_seconds, repeat=True)

    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
    if hang_trace_seconds > 0:
        faulthandler.cancel_dump_traceback_later()
