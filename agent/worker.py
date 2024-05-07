from Agent import Agent
from Watchdog import Watchdog
from RatchetEnvironment import RatchetEnvironment
from Buffer import Transition, TransitionMessage

import pickle
import numpy as np

from redis import Redis
from redis import from_url as redis_from_url

from PPOAgent import PPOAgent

import torch


features = 18 + 128
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

    # Make new environment and watchdog
    env = RatchetEnvironment(process_name=process_name, eval_mode=eval_mode)
    watchdog = Watchdog(env.game, rpcs3_path=rpcs3_path, process_name=process_name, render=render)

    # Randomized worker ID
    worker_id = np.random.randint(0, 999999)
    worker_id = f"worker-{worker_id}"

    # Watchdog starts RPCS3 and the game for us if it's not already running
    watchdog.start()
    env.start()

    # Connect to Redis
    redis = redis_from_url(f"redis://{args.redis_host}:{args.redis_port}")

    if eval_mode:
        # Draw a visualization of the actions and other information
        from Visualizer import draw_bars

    # Agent that we will use only for inference, learning related parameters are not used
    agent = PPOAgent(features, 7, 0.00003, 0.0001, 0.99, 80, 0.2)

    last_model_fetch_time = 0

    total_steps = 0
    episodes = 0
    scores = []

    # Start stepping through the environment
    while True:
        model_timestamp = redis.get("rac1.fitness-course.model_timestamp")
        if model_timestamp is not None and float(model_timestamp) > last_model_fetch_time:
            # Load the latest model from Redis
            configuration["model"] = redis.get("rac1.fitness-course.model")
            if configuration["model"] is not None:
                agent.load_policy_dict(pickle.loads(configuration["model"]))
                last_model_fetch_time = float(redis.get("rac1.fitness-course.model_timestamp"))

        agent.start_new_episode()
        state, _, _ = env.reset()

        state_sequence = np.zeros((sequence_length, features), dtype=np.float32)
        state_sequence[-1] = state

        accumulated_reward = 0
        steps = 0

        last_done = True

        while True:
            actions, logprob, state_value, mu, log_std = agent.choose_action(state_sequence)
            actions = actions.squeeze().cpu()

            new_state, reward, done = env.step(actions)

            time_left = (30 * 30 - env.time_since_last_checkpoint) / 30

            if not eval_mode:
                if total_steps > 128:
                    transition = Transition(state_sequence, actions, reward, last_done, logprob, state_value,
                                            mu, log_std, None, None)
                    message = TransitionMessage(transition, worker_id)

                    # Pickle the transition and publish it to the "replay_buffer" channel
                    data = pickle.dumps(message)
                    redis.publish("rac1.fitness-course.replay_buffer", data)
            else:
                # Visualize the actions
                draw_bars(actions, state_value, time_left/30)

                if done:
                    break

            # Roll the state sequence and append the new normalized state
            new_state_sequence = np.roll(state_sequence, -1, axis=0)
            new_state_sequence[-1] = new_state

            state_sequence = new_state_sequence
            last_done = done

            accumulated_reward += reward
            steps += 1
            total_steps += 1

            if steps % 5 == 0:
                print(f"Score: %6.2f    death: %05.2f checkpoint: %d  closest_dist: %02.2f  value: %3.2f         " % (
                    accumulated_reward,
                    time_left,
                    env.n_checkpoints,
                    env.closest_distance_to_checkpoint,
                    state_value.item() if state_value is not None else 0.0
                ), end="\r")

                model_timestamp = redis.get("rac1.fitness-course.model_timestamp")
                if model_timestamp is not None and float(model_timestamp) > last_model_fetch_time:
                    # Load the latest model from Redis
                    configuration["model"] = redis.get("rac1.fitness-course.model")
                    if configuration["model"] is not None:
                        agent.load_policy_dict(pickle.loads(configuration["model"]))
                        last_model_fetch_time = float(redis.get("rac1.fitness-course.model_timestamp"))

            if done:
                break

        scores.append(accumulated_reward)
        avg_score = np.mean(scores[-100:])

        print('steps %d' % total_steps, 'score: %.2f' % accumulated_reward, 'checkpoints: %d' % env.n_checkpoints,
              'avg score: %.2f' % avg_score, 'chkpt update time: %.0f' % last_model_fetch_time)
              # 'eps: %.2f' % agent.epsilon if agent.epsilon > agent.eps_min else '')

        # Append score to Redis key "scores"
        if not eval_mode:
            redis.rpush("rac1.fitness-course.avg_scores", accumulated_reward)
            redis.rpush("rac1.fitness-course.checkpoints", env.n_checkpoints)

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
        parser.add_argument("--eval", type=bool, default=False)

        args = parser.parse_args()

        start_worker(args)
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
