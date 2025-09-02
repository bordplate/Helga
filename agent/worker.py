import time

import torch

from Environments.CaptureTheFlagEnvironment import CaptureTheFlagEnvironment
from RunningStats import RunningStats
from VectorizedStats import VecNormalize
from Watchdog import Watchdog

import numpy as np

from PPO.PPOAgent import PPOAgent

from RedisHub import RedisHub

features = 43
sequence_length = 1

configuration = {
    "model": None,
    "epsilon": 0.005,
    "min_epsilon": 0.005,
}


def start_worker(args):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()

    # Get paths from arguments
    rpcs3_path = args.rpcs3_path
    process_name = args.process_name
    render = args.render
    eval_mode = args.eval
    project_key = args.project_key
    device = "cpu" if args.cpu_only else device

    best_checkpoint = 0

    watchdog = None

    # If we're not being debugged in PyCharm mode, we start a new RPCS3 process using the watchdog, otherwise we connect to an existing one
    pid = 0
    import sys
    if not "pydevd" in sys.modules:
        # Make new environment and watchdog
        watchdog = Watchdog(render=eval_mode)
        if not watchdog.start(force=True):
            print("Damn, watchdog failed to start the process!")
            exit(-1)
        else:
            pid = watchdog.pid

        time.sleep(5)
    else:
        # Find the rpcs3 process
        import psutil
        for proc in reversed(list(psutil.process_iter())):
            if proc.name() == "rpcs3":
                pid = proc.pid
                break

    env = CaptureTheFlagEnvironment(pid)

    # Watchdog starts RPCS3 and the game for us if it's not already running
    env.start()

    if not watchdog is None:
        watchdog.watch(env)

    # Connect to Redis
    redis = RedisHub(f"redis://{args.redis_host}:{args.redis_port}", f"{project_key}.rollout_buffer", device=device)

    state_running_stats = VecNormalize(shape=(sequence_length, features), device=device)

    # Agent that we will use only for inference, learning related parameters are not used
    agent = PPOAgent(features, 6, log_std=-0.5, device=device)

    if eval_mode:
        agent.policy.actor.max_log_std = 0.8
    else:
        agent.policy.actor.max_log_std = 0.8

    total_steps = 0
    episodes = 0
    scores = []

    all_zeros = torch.zeros((features), dtype=torch.bfloat16).to(device)

    # Start stepping through the environment
    while True:
        torch.cuda.empty_cache()

        states = []

        player_states = env.reset()
        for state in player_states:
            states += [state[0].to(device)]

        rewards = [0.0, 0.0, 0.0, 0.0]

        # state_running_stats.update(state)
        # state = state_running_stats.normalize_observation(state)

        # state_sequence = torch.zeros((sequence_length, features), dtype=torch.bfloat16).to(device)
        # state_sequence[-1] = state

        accumulated_reward = 0
        steps = 0

        must_check_new_model = False

        agent.action_mask = redis.get_action_mask()

        start_time = time.time()

        while True:
            # while redis.check_buffer_full():
            #     must_check_new_model = True
            #     time.sleep(0.1)

            if steps % 5 == 0 or must_check_new_model:
                new_model = redis.get_new_model()
                if new_model is not None:
                    agent.load_policy_dict(new_model)

            player_actions = []
            player_logprobs = []
            player_values = []
            for player_id in range(4):
                actions, logprob, state_value = agent.choose_action(states[player_id])
                actions = actions.to(dtype=torch.float32).squeeze().cpu()
                player_actions += [actions]
                player_logprobs += [logprob]
                player_values += [state_value]

            new_states = env.step(player_actions)

            states = []

            done = False

            for player_id in range(4):
                state, reward, terminal = new_states[player_id]
                #if terminal and not eval_mode:
                if terminal:
                    done = True

                rewards[player_id] += reward

                if not eval_mode:
                    redis.add(player_id, state, player_actions[player_id], reward, done, player_logprobs[player_id], player_values[player_id])

                states += [state.to(device)]

            steps += 1
            total_steps += 1

            if eval_mode or steps % 5 == 0:
                steps_per_second = steps / (time.time() - start_time)

                print(f"\rP1: %6.2f; P2: %6.2f; P3: %6.2f; P4: %6.2f    sps: %3.2f; timeout: %.2f         " % (
                    rewards[0], rewards[1], rewards[2], rewards[3],
                    steps_per_second,
                    env.timeout / 30
                ), end="")

            if done:
                break

        scores.append(rewards)
        avg_score = np.mean(scores[-100:])

        print(f"\r==P1: %6.2f; P2: %6.2f; P3: %6.2f; P4: %6.2f;  avg score: %.2f; steps: %d" %
          (rewards[0], rewards[1], rewards[2], rewards[3], avg_score, steps)
        )

        # Append score to Redis key "scores"
        if not eval_mode:
            redis.redis.rpush(f"{project_key}.avg_scores", rewards[0])
            redis.redis.rpush(f"{project_key}.episode_length", steps)

        episodes += 1


if __name__ == "__main__":
    # Catch Ctrl+C and exit gracefully
    args = None
    try:
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--rpcs3-path", type=str, required=False)
        parser.add_argument("--process-name", type=str, default="rpcs3")
        parser.add_argument("--redis-host", type=str, default="localhost")
        parser.add_argument("--redis-port", type=int, default=6379)
        parser.add_argument("--render", action="store_true", default=True)
        parser.add_argument("--force-watchdog", action="store_false")
        parser.add_argument("--eval", type=bool, action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--project-key", type=str, default="rac3.ctf")
        parser.add_argument("--cpu-only", action="store_true", default=False)

        args = parser.parse_args()

        with torch.no_grad() and torch.autograd.no_grad() and torch.inference_mode():
            start_worker(args)
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
    except ValueError:
        start_worker(args)
    # except OSError:
    #     start_worker(args)
