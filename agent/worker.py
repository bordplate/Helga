from Agent import Agent
from Watchdog import Watchdog
from RatchetEnvironment import RatchetEnvironment
from ReplayBuffer import Transition, TransitionMessage

import pickle
import numpy as np

from redis import Redis
from redis import from_url as redis_from_url

from PPOAgent import PPOAgent


features = 15 + 128
sequence_length = 8

configuration = {
    "model": None,
    "epsilon": 0.005,
    "min_epsilon": 0.005,
}


def update_configuration(redis: Redis):
    configuration["action_std"] = float(redis.get("rac1.fitness-course.action_std")) if redis.get("rac1.fitness-course.action_std") is not None else 1.0
    configuration["min_epsilon"] = float(redis.get("rac1.fitness-course.min_epsilon")) if redis.get("rac1.fitness-course.min_epsilon") is not None else 0.005


def start_worker():
    # Get paths from arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpcs3-path", type=str, required=True)
    parser.add_argument("--process-name", type=str, required=True)
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--force-watchdog", action="store_false")
    parser.add_argument("--epsilon", type=float, default=None)
    args = parser.parse_args()

    rpcs3_path = args.rpcs3_path
    process_name = args.process_name
    render = args.render
    epsilon_override = args.epsilon

    # Make new environment and watchdog
    env = RatchetEnvironment(process_name=process_name)
    watchdog = Watchdog(env.game, rpcs3_path=rpcs3_path, process_name=process_name, render=render)

    # Randomized worker ID
    worker_id = np.random.randint(0, 999999)
    worker_id = f"worker-{worker_id}"

    # Watchdog starts RPCS3 and the game for us if it's not already running
    watchdog.start()
    env.start()

    # Connect to Redis
    redis = redis_from_url(f"redis://{args.redis_host}:{args.redis_port}")

    if epsilon_override is None:
        update_configuration(redis)
    else:
        print("Running with epsilon override:", epsilon_override)
        configuration["action_std"] = float(epsilon_override)
        configuration["min_action_std"] = float(epsilon_override)

    # Agent that we will use only for inference
    # agent = PPOAgent(gamma=0.99, epsilon=configuration["epsilon"], batch_size=0, n_actions=13, eps_end=configuration["min_epsilon"],
    #               input_dims=features, lr=0, sequence_length=8)
    agent = PPOAgent(features, 7, 0.00003, 0.0001, 0.99, 80, 0.2)
    agent.policy.eval()
    agent.policy_old.eval()

    last_model_fetch_time = 0

    total_steps = 0
    episodes = 0
    scores = []
    losses = []

    # Start stepping through the environment
    while True:
        model_timestamp = redis.get("rac1.fitness-course.model_timestamp")
        if model_timestamp is not None and float(model_timestamp) > last_model_fetch_time:
            # Load the latest model from Redis
            configuration["model"] = redis.get("rac1.fitness-course.model")
            if configuration["model"] is not None:
                agent.load_policy_dict(pickle.loads(configuration["model"]))
                last_model_fetch_time = float(redis.get("rac1.fitness-course.model_timestamp"))

        if epsilon_override is None:
            update_configuration(redis)
            agent.set_action_std(configuration["action_std"])
            agent.eps_min = configuration["min_epsilon"]
        else:
            agent.set_action_std(float(epsilon_override))
            agent.eps_min = float(epsilon_override)

        agent.start_new_episode()
        state, _, _ = env.reset()

        state_sequence = np.zeros((sequence_length, features), dtype=np.float32)
        state_sequence[-1] = state

        accumulated_reward = 0
        steps = 0

        while True:
            hidden_state, cell_state = agent.policy_old.hidden_state.detach(), agent.policy_old.cell_state.detach()
            #hidden_state, cell_state = None, None

            actions, _, logprob, state_value = agent.choose_action(state_sequence)
            state, reward, done = env.step(actions)

            new_state_sequence = np.concatenate((state_sequence[1:], [state]))

            # agent.replay_buffer.add(state_sequence, actions, reward, new_state_sequence, done, logprob, state_value, None, None)

            # if total_steps > 0 and total_steps % 500 == 0:
            #     loss = agent.learn()
            #     losses.append(loss)
            #
            # if total_steps > 0 and total_steps % action_std_decay_freq == 0:
            #     agent.decay_action_std(action_std_decay_rate, min_action_std)

            if epsilon_override is not None:
                transition = Transition(state_sequence, actions, reward, done, logprob, state_value,
                                        hidden_state, cell_state)
                message = TransitionMessage(transition, worker_id)

                # Pickle the transition and publish it to the "replay_buffer" channel
                data = pickle.dumps(message)
                redis.publish("rac1.fitness-course.replay_buffer", data)

            state_sequence = new_state_sequence

            accumulated_reward += reward
            steps += 1
            total_steps += 1

            if steps % 5 == 0:
                time_left = (30 * 30 - env.time_since_last_checkpoint) / 30
                print(f"Score: %6.2f    death: %05.2f checkpoint: %d  closest_dist: %f         " % (
                    accumulated_reward,
                    time_left,
                    env.n_checkpoints,
                    env.closest_distance_to_checkpoint
                ), end="\r")

                # When time left is less than 5 seconds, we increase epsilon as a last ditch effort to explore
                if epsilon_override is None and time_left < 5:
                    agent.epsilon = 0.25
                # elif agent.epsilon == 0.25 and time_left > 5:  # Reset epsilon to normal if agent hit checkpoint and gained more time
                #     agent.epsilon = configuration["epsilon"]

            if done:
                break

        scores.append(accumulated_reward)
        avg_score = np.mean(scores[-100:])

        print('steps %d' % total_steps, 'score: %.2f' % accumulated_reward, 'checkpoints: %d' % env.n_checkpoints,
              'avg score: %.2f' % avg_score, 'chkpt update time: %.0f' % last_model_fetch_time,
              'action_std: %.2f' % agent.action_std)
              # 'eps: %.2f' % agent.epsilon if agent.epsilon > agent.eps_min else '')

        # Append score to Redis key "scores"
        if epsilon_override is None:
            redis.rpush("rac1.fitness-course.avg_scores", accumulated_reward)

            redis.rpush("rac1.fitness-course.checkpoints", env.n_checkpoints)

        episodes += 1


if __name__ == "__main__":
    # Catch Ctrl+C and exit gracefully
    try:
        start_worker()
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
