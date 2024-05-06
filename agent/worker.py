from Agent import Agent
from Watchdog import Watchdog
from RatchetEnvironment import RatchetEnvironment
from Buffer import Transition, TransitionMessage

import pickle
import numpy as np

from redis import Redis
from redis import from_url as redis_from_url

from PPOAgent import PPOAgent

import pygame
import torch


features = 18 + 128
sequence_length = 1

configuration = {
    "model": None,
    "epsilon": 0.005,
    "min_epsilon": 0.005,
}

# Pygame setup
screen_width, screen_height = 1000, 220

bar_width = 80
spacing = 20

color_active = (0, 255, 0)
color_inactive = (255, 0, 0)

# colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (75, 0, 130)]
labels = ["LJoyX", "LJoyY", "RJoyX", "RJoyY", "R1", "Cross", "Square"]
# font = pygame.font.Font(None, 24)
global font


def draw_bars(screen, actions, state_value, progress):
    screen.fill((0, 0, 0))
    for i, action in enumerate(actions):
        color = color_active if action > 0.5 else color_inactive

        if i < 4:
            color = color_active if abs(action) > 0.2 else color_inactive

        height = int(abs(action) * 90)  # Scale action value to height
        bar_x = i * (bar_width + spacing) + 50
        bar_y = screen_height - 40 - height

        if action < 0:
            bar_y += height

        pygame.draw.rect(screen, color, pygame.Rect(bar_x, bar_y - 90, bar_width, height))

        # pygame.draw.rect(screen, color, pygame.Rect(bar_x, screen_height - 20 - height, bar_width, height))

        label = font.render(labels[i], True, (255, 255, 255))
        label_pos_x = bar_x + (bar_width - label.get_width()) // 2  # Calculate x position to center the label
        screen.blit(label, (label_pos_x, screen_height - 40))

    # Draw state_value bar and label, (state_value is between -1 and 1)
    _state_value = (state_value + 1) / 2
    _state_value = max(0, min(1, _state_value))
    height = int(abs(state_value) * 90)
    bar_x = 7 * (bar_width + spacing) + 50
    bar_y = screen_height - 40 - height

    if state_value < 0:
        bar_y += height

        # Gradient red to white
        col = (int(255 * _state_value), int(255 * (1 - _state_value)), int(255 * (1 - _state_value)))
    else:
        # Gradient green to white
        col = (int(255 * (1 - _state_value)), int(255 * _state_value), int(255 * (1 - _state_value)))

    pygame.draw.rect(screen, col, pygame.Rect(bar_x, bar_y - 90, bar_width, height))

    label = font.render("State Value", True, (255, 255, 255))
    label_pos_x = bar_x + (bar_width - label.get_width()) // 2
    screen.blit(label, (label_pos_x, screen_height - 40))

    # Draw a progress bar at the bottom of the screen
    progress_bar_width = int(progress * screen_width)
    pygame.draw.rect(screen, (255, 0, 0xdb), pygame.Rect(0, screen_height - 10, progress_bar_width, 10))

    pygame.display.flip()


def update_configuration(redis: Redis):
    configuration["action_std"] = float(redis.get("rac1.fitness-course.action_std")) if redis.get("rac1.fitness-course.action_std") is not None else 1.0
    configuration["min_epsilon"] = float(redis.get("rac1.fitness-course.min_epsilon")) if redis.get("rac1.fitness-course.min_epsilon") is not None else 0.005


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.run_var = None

        self.n_rewards = 0
        self.mean_rewards = None
        self.run_var_rewards = None

    def update(self, x, reward=0.0):
        x = np.array(x)  # Ensure x is an array
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.run_var = np.zeros_like(x)

        self.n += 1
        old_mean = np.copy(self.mean)
        self.mean += (x - self.mean) / self.n
        self.run_var += (x - old_mean) * (x - self.mean)

        if reward is not None:
            if self.mean_rewards is None:
                self.mean_rewards = 0.0
                self.run_var_rewards = 0.0

            self.n_rewards += 1
            old_mean_rewards = self.mean_rewards
            self.mean_rewards += (reward - self.mean_rewards) / self.n_rewards
            self.run_var_rewards += (reward - old_mean_rewards) * (reward - self.mean_rewards)

    def variance(self):
        return self.run_var / self.n if self.n > 1 else np.zeros_like(self.run_var)

    def standard_deviation(self):
        return np.sqrt(self.variance())

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
        # Pygame initialization for visualization
        pygame.init()
        screen_width, screen_height = 1000, 220
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Action Visualization")
        globals()["font"] = pygame.font.Font(None, 24)

        print("Running with epsilon override:", epsilon_override)
        configuration["action_std"] = float(epsilon_override)
        configuration["min_action_std"] = float(epsilon_override)

    # Agent that we will use only for inference
    # agent = PPOAgent(gamma=0.99, epsilon=configuration["epsilon"], batch_size=0, n_actions=13, eps_end=configuration["min_epsilon"],
    #               input_dims=features, lr=0, sequence_length=8)
    agent = PPOAgent(features, 7, 0.00003, 0.0001, 0.99, 80, 0.2)

    last_model_fetch_time = 0

    total_steps = 0
    episodes = 0
    scores = []
    losses = []

    max_ep_len = 1024 * 8

    state_stats = RunningStats()

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
            # agent.set_action_std(configuration["action_std"])
            agent.eps_min = configuration["min_epsilon"]
        else:
            # agent.set_action_std(float(epsilon_override))
            agent.eps_min = float(epsilon_override)

        agent.start_new_episode()
        state, _, _ = env.reset()

        state_sequence = np.zeros((sequence_length, features), dtype=np.float32)

        # for i in range(sequence_length):
        #     state_stats.update(state)
        #     normalized_state = (state - state_stats.mean) / (state_stats.standard_deviation() + 1e-8)
        #     state_sequence[i] = normalized_state
        #     if i < sequence_length - 1:
        #         state, _, _ = env.step(np.zeros(agent.policy.action_dim))

        state_sequence[-1] = state

        accumulated_reward = 0
        steps = 0

        hidden_state, cell_state = None, None

        last_done = True

        while True:
            hidden_state, cell_state = agent.policy.hidden_state.detach(), agent.policy.cell_state.detach()

            actions, logprob, state_value, mu, log_std = agent.choose_action(state_sequence)
            actions = actions.squeeze().cpu()

            new_state, reward, done = env.step(actions)

            time_left = (30 * 30 - env.time_since_last_checkpoint) / 30
            # agent.replay_buffer.add(state_sequence, actions, reward, new_state_sequence, done, logprob, state_value, None, None)

            # if total_steps > 0 and total_steps % 500 == 0:
            #     loss = agent.learn()
            #     losses.append(loss)
            #
            # if total_steps > 0 and total_steps % action_std_decay_freq == 0:
            #     agent.decay_action_std(action_std_decay_rate, min_action_std)

            if epsilon_override is None:
                if total_steps > 128:
                    transition = Transition(state_sequence, actions, reward, last_done, logprob, state_value,
                                            mu, log_std, hidden_state, cell_state)
                    message = TransitionMessage(transition, worker_id)

                    # Pickle the transition and publish it to the "replay_buffer" channel
                    data = pickle.dumps(message)
                    redis.publish("rac1.fitness-course.replay_buffer", data)
            else:
                # Visualize the actions
                draw_bars(screen, actions, state_value, time_left/30)

                # Pygame event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                if done:
                    break

            state_stats.update(new_state, reward)
            normalized_new_state = (new_state - state_stats.mean) / (state_stats.standard_deviation() + 1e-8)

            # Roll the state sequence and append the new normalized state
            new_state_sequence = np.roll(state_sequence, -1, axis=0)
            # new_state_sequence[-1] = normalized_new_state
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

                if epsilon_override is None:
                    update_configuration(redis)
                    # agent.set_action_std(configuration["action_std"])
                    agent.eps_min = configuration["min_epsilon"]
                else:
                    # agent.set_action_std(float(epsilon_override))
                    agent.eps_min = float(epsilon_override)

                # When time left is less than 5 seconds, we increase epsilon as a last ditch effort to explore
                # if epsilon_override is None and time_left < 5:
                #     agent.epsilon = 0.25
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
