"""Evaluate FastWAM on LIBERO-plus through a remote FastWAM policy server."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import time

import numpy as np
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastwam_libero_plus_mpl_cache")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.libero_plus.fastwam_client import FastWAMClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def get_env(task, resolution, seed):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env


def get_proprio(obs):
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def convert_action(action, binarize_gripper=True):
    action = np.asarray(action, dtype=np.float32).copy()
    # FastWAM's training processor uses [0, 1] for gripper; LIBERO expects
    # -1=open, +1=close, with the simulator adapter applying the final sign.
    action[..., -1] = action[..., -1] * 2.0 - 1.0
    action[..., -1] *= -1.0
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    return action


def max_steps_for_suite(suite):
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }[suite]


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def evaluate(args):
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    start = 0 if args.start_idx < 0 else args.start_idx
    end = suite.n_tasks if args.end_idx < 0 else args.end_idx
    if not (0 <= start < end <= suite.n_tasks):
        raise ValueError(f"Invalid task range [{start}, {end}) for {args.task_suite_name} ({suite.n_tasks})")

    mapping_path = pathlib.Path(args.libero_home) / "libero/libero/benchmark/task_classification.json"
    with mapping_path.open(encoding="utf-8") as f:
        classification = json.load(f)[args.task_suite_name]
    categories = {}
    for item in classification[start:end]:
        category = item["category"]
        categories.setdefault(category, {"total_count": 0, "success_count": 0})["total_count"] += args.num_trials

    client = FastWAMClient(args.host, args.port)
    logging.info("Connected to FastWAM server: metadata=%s", client.metadata)
    suite_steps = max_steps_for_suite(args.task_suite_name)
    total_episodes = total_successes = 0
    total_tasks = end - start
    target_episodes = total_tasks * args.num_trials
    eval_start = time.monotonic()
    logging.info(
        "Evaluation started: suite=%s tasks=%d [%d,%d) trials_per_task=%d target_episodes=%d",
        args.task_suite_name,
        total_tasks,
        start,
        end,
        args.num_trials,
        target_episodes,
    )
    pathlib.Path(args.log_path).mkdir(parents=True, exist_ok=True)

    try:
        for task_id in range(start, end):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            env = get_env(task, 256, args.seed)
            task_successes = 0
            try:
                for episode_idx in range(args.num_trials):
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_idx])
                    pending = []
                    done = False
                    t = 0
                    while t < suite_steps + args.num_steps_wait:
                        if t < args.num_steps_wait:
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue
                        if not pending:
                            primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            action_chunk = client.predict_action(
                                primary=primary,
                                wrist=wrist,
                                proprio=get_proprio(obs),
                                instruction=task.language,
                            )
                            pending = convert_action(action_chunk, args.binarize_gripper)[: args.replan_steps].tolist()
                        obs, _, done, _ = env.step(pending.pop(0))
                        if done:
                            break
                        t += 1

                    total_episodes += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                    category = classification[task_id]["category"]
                    categories[category]["success_count"] += int(done)
                    elapsed = time.monotonic() - eval_start
                    episodes_per_second = total_episodes / max(elapsed, 1e-6)
                    remaining_episodes = target_episodes - total_episodes
                    eta = remaining_episodes / max(episodes_per_second, 1e-9)
                    logging.info(
                        "%s task=%d episode=%d success=%s | tasks=%d/%d episodes=%d/%d "
                        "remaining=%d successes=%d total_success_rate=%.2f%% elapsed=%s ETA=%s",
                        args.task_suite_name,
                        task_id,
                        episode_idx,
                        done,
                        task_id - start + int(episode_idx + 1 == args.num_trials),
                        total_tasks,
                        total_episodes,
                        target_episodes,
                        remaining_episodes,
                        total_successes,
                        total_successes / total_episodes * 100.0,
                        format_duration(elapsed),
                        format_duration(eta),
                    )
                logging.info(
                    "Task %d finished: success_rate=%.2f%% (%d/%d) | total_success_rate=%.2f%% | ETA=%s",
                    task_id,
                    task_successes / args.num_trials * 100.0,
                    task_successes,
                    args.num_trials,
                    total_successes / total_episodes * 100.0,
                    format_duration((target_episodes - total_episodes) / max(total_episodes / max(time.monotonic() - eval_start, 1e-6), 1e-9)),
                )
            finally:
                env.close()
    finally:
        client.close()

    result_name = f"{args.task_suite_name}.json" if start == 0 and end == suite.n_tasks else f"{args.task_suite_name}_{start}_{end}.json"
    output = {key: {**value, "success_rate": value["success_count"] / value["total_count"]} for key, value in categories.items()}
    with (pathlib.Path(args.log_path) / result_name).open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logging.info(
        "Finished %s [%d,%d): episodes=%d/%d successes=%d total_success_rate=%.2f%% elapsed=%s",
        args.task_suite_name,
        start,
        end,
        total_episodes,
        target_episodes,
        total_successes,
        total_successes / max(total_episodes, 1) * 100.0,
        format_duration(time.monotonic() - eval_start),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-home", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-suite-name", required=True)
    parser.add_argument("--start-idx", type=int, default=-1)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--binarize-gripper", action=argparse.BooleanOptionalAction, default=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", force=True)
    main()
