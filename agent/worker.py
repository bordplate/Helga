import time

import torch

from Watchdog import Watchdog
from Environments.FitnessCourseEnvironment import FitnessCourseEnvironment

import numpy as np

from PPO.PPOAgent import PPOAgent

from RedisHub import RedisHub

features = 23 + 128
sequence_length = 8

configuration = {
    "model": None,
    "epsilon": 0.005,
    "min_epsilon": 0.005,
}


def start_worker(args):
    # Get paths from arguments
    rpcs3_path = args.rpcs3_path
    process_name = args.process_name
    render = args.render
    eval_mode = args.eval
    project_key = args.project_key

    # Make new environment and watchdog
    env = FitnessCourseEnvironment(process_name=process_name, eval_mode=eval_mode)
    watchdog = Watchdog(env.game, rpcs3_path=rpcs3_path, process_name=process_name, render=render)

    # Watchdog starts RPCS3 and the game for us if it's not already running
    watchdog.start()
    env.start()

    # Connect to Redis
    redis = RedisHub(f"redis://{args.redis_host}:{args.redis_port}", f"{project_key}.rollout_buffer")

    if eval_mode:
        # Draws a visualization of the actions and other information
        import Visualizer
        import Plotter
        Plotter.start_plotting()

    # Agent that we will use only for inference, learning related parameters are not used
    agent = PPOAgent(features, 7, log_std=-0.5)

    total_steps = 0
    episodes = 0
    scores = []

    # Start stepping through the environment
    while True:
        agent.start_new_episode()
        state, _, _ = env.reset()

        state_sequence = np.zeros((sequence_length, features), dtype=np.float32)
        state_sequence[-1] = state

        accumulated_reward = 0
        steps = 0

        last_done = True

        must_check_new_model = False

        while True:
            while redis.check_buffer_full():
                must_check_new_model = True
                time.sleep(0.1)

            if steps % 5 == 0 or must_check_new_model:
                new_model = redis.get_new_model()
                if new_model is not None:
                    agent.load_policy_dict(new_model)

            actions, logprob, state_value, y_t = agent.choose_action(state_sequence)
            actions = actions.squeeze().cpu()

            new_state, reward, done = env.step(actions)

            time_left = (30 * 30 - env.time_since_last_checkpoint) / 30

            if not eval_mode:
                redis.add(state_sequence, actions, reward, last_done, logprob, state_value, y_t)
            else:
                # Visualize the actions
                Visualizer.draw_state_value_face(state_value)
                Visualizer.draw_bars(actions, state_value, time_left/30)

                Plotter.add_data(state_value.item(), reward)

            # Roll the state sequence and append the new normalized state
            new_state_sequence = np.roll(state_sequence, -1, axis=0)
            new_state_sequence[-1] = new_state

            state_sequence = new_state_sequence
            last_done = done

            accumulated_reward += reward
            steps += 1
            total_steps += 1

            if eval_mode or steps % 5 == 0:
                print(f"Score: %6.2f    death: %05.2f checkpoint: %d  closest_dist: %02.2f  value: %3.2f         " % (
                    accumulated_reward,
                    time_left,
                    env.n_checkpoints,
                    env.closest_distance_to_checkpoint,
                    state_value.item() if state_value is not None else 0.0
                ), end="\r")

            if done:
                break

        scores.append(accumulated_reward)
        avg_score = np.mean(scores[-100:])

        print('score: %.2f' % accumulated_reward, 'checkpoints: %d' % env.n_checkpoints,
              'avg score: %.2f' % avg_score)

        # Append score to Redis key "scores"
        if not eval_mode:
            redis.redis.rpush(f"{project_key}.avg_scores", accumulated_reward)
            redis.redis.rpush(f"{project_key}.checkpoints", env.n_checkpoints)

        episodes += 1


if __name__ == "__main__":
    # Catch Ctrl+C and exit gracefully
    try:
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--rpcs3-path", type=str, required=True)
        parser.add_argument("--process-name", type=str, required=True)
        parser.add_argument("--redis-host", type=str, default="localhost")
        parser.add_argument("--redis-port", type=int, default=6379)
        parser.add_argument("--render", action="store_true", default=True)
        parser.add_argument("--force-watchdog", action="store_false")
        parser.add_argument("--eval", type=bool, action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--project-key", type=str, default="rac1.fitness-course")

        args = parser.parse_args()

        with torch.no_grad():
            start_worker(args)
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
