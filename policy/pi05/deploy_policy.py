import numpy as np
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    from pi_model import PI0

    train_config_name, model_name, checkpoint_id, pi0_step, num_inference_steps = (
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args["checkpoint_id"],
        usr_args["pi0_step"],
        usr_args["num_inference_steps"],
    )
    return PI0(
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        num_inference_steps,
    )


def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()
    if hasattr(model, "call"):
        input_rgb_arr, input_state = encode_obs(observation)
        payload = {
            "images": {
                "head_camera": input_rgb_arr[0],
                "right_camera": input_rgb_arr[1],
                "left_camera": input_rgb_arr[2],
            },
            "state": input_state,
            "instruction": instruction,
        }
        actions = np.asarray(model.call(func_name="act", obs=payload))
    else:
        if model.observation_window is None:
            model.set_language(instruction)
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
        actions = model.get_action()[:model.pi0_step]

    replan_index = int(getattr(model, "_arm_rollout_replan_index", 0))

    for chunk_action_index, action in enumerate(actions):
        TASK_ENV._arm_replan_index = replan_index
        TASK_ENV._arm_chunk_action_index = chunk_action_index
        TASK_ENV.take_action(action)
        if TASK_ENV.eval_success:
            break
    model._arm_rollout_replan_index = replan_index + 1

    # The outer evaluator refreshes the observation immediately before the
    # next policy call.  Reading all three RT cameras after every open-loop
    # waypoint only overwrote this single-frame buffer with values that were
    # never consumed, multiplying camera.get_picture calls by the chunk size.

    # ============================


def probe(TASK_ENV, model, observation, *, noise_seed):
    """Capture initial solver features without calling TASK_ENV.take_action."""
    if model.observation_window is None:
        model.set_language(TASK_ENV.get_instruction())

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    return model.get_solver_probe(noise_seed)


def reset_model(model):
    if hasattr(model, "call"):
        return model.call(func_name="reset_model")
    return model.reset_model()
