from agent import Agent
from ratchet_environment import RatchetEnvironment
from watchdog import Watchdog

import numpy as np
import wandb

import os
import random
import torch

import sys


enable_wandb = "pydevd" not in sys.modules  # Disable wandb if debugging

load_model = ""  # Set to model filename to load, empty string to start fresh


def start_training():
    env = RatchetEnvironment()
    env.start()

    # Watchdog to restart the environment if it crashes or stalls
    watchdog = Watchdog(env.game)
    watchdog.start()

    random_instance_id = random.randint(0, 999999)  # Generate a random instance ID to use when saving the model

    sequence_length = 30  # Number of observations to use as input to the network

    learning_rate = 3.5e-4
    features = 23
    batch_size = 64
    train_frequency = 4
    target_update_frequency = 10000  # How often we update the target network

    steps_since_update = 0

    if enable_wandb:
        wandb.init(project="rac1-hoverboard", config={
            "learning_rate": learning_rate,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "features": features,
            "train_frequency": train_frequency,
            "target_update_frequency": target_update_frequency,
        })

        update_graph_html(wandb.run.get_url())

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=batch_size, n_actions=4, eps_end=0.005,
                  input_dims=features, lr=learning_rate, sequence_length=sequence_length)

    # Load existing model if load_model is set
    if load_model:
        model_path = os.path.join('models_bak', load_model)
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path)
            agent.Q_eval.load_state_dict(checkpoint['model_state_dict'])
            agent.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = checkpoint['epsilon']

            agent.Q_eval = agent.Q_eval.to(agent.Q_eval.device)

            agent.Q_eval.train()
        else:
            print(f"Model file {model_path} not found. Starting fresh.")

    scores, eps_history = [], []
    n_games = 1000000

    print(f"Device: {agent.Q_eval.device}. Instance ID: {random_instance_id}")

    furthest_distance = 0.0

    total_steps = 0

    for i in range(n_games):
        done = False

        observation = env.reset()[0]

        # Reinitialize the observation_sequence for the new episode
        observation_sequence = np.zeros((sequence_length, features))  # Reset the sequence

        # Set the last observation to the initial observation from the environment
        observation_sequence[-1] = observation

        accumulated_reward = 0

        losses = []

        steps = 0

        agent.start_new_episode()

        while not done:
            action = agent.choose_action(observation_sequence)
            observation_, reward, done = env.step(action)

            # Frame skip to improve temporal resolution
            new_observation_sequence = np.concatenate((observation_sequence[1:], np.array([observation_])), axis=0)
            agent.store_transition(observation_sequence, action, reward, new_observation_sequence, done)

            observation_sequence = new_observation_sequence

            steps_since_update += 1

            accumulated_reward += reward

            steps += 1
            total_steps += 1

            if steps % train_frequency == 0:
                loss = agent.learn()
                losses.append(loss)

        scores.append(accumulated_reward)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        loss = agent.learn(num_batches=1, terminal_learn=True, average_reward=avg_score)
        losses.append(loss)

        if steps_since_update > target_update_frequency:
            agent.update_target_network()
            steps_since_update = 0

        if env.distance_traveled > furthest_distance:
            furthest_distance = env.distance_traveled
            print(f"New furthest distance: {furthest_distance}")

        # Save the model every 200 episodes
        if i % 200 == 0 and i != 0:
            model_filename = f"rac1_hoverboard_{random_instance_id}_{i}.pt"
            model_path = os.path.join('models_bak', model_filename)
            if not os.path.exists('models_bak'):
                os.makedirs('models_bak')
            agent.Q_eval.train()
            torch.save({
                'model_state_dict': agent.Q_eval.state_dict(),
                'optimizer_state_dict': agent.Q_eval.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'episode': i,
            }, model_path)
            print(f"Model saved as {model_path}")

        # Get the current learning rate
        for n, g in enumerate(agent.Q_eval.optimizer.param_groups):
            if n > 1:
                print(f"WARNING: More than one optimizer param group found. This is probably a bug.")
                break

            learning_rate = g['lr']

        # Logging

        if enable_wandb:
            wandb.log({"reward": accumulated_reward, "epsilon": agent.epsilon, "average_reward": avg_score,
                       "distance_traveled": env.distance_traveled, "furthest_distance": furthest_distance,
                       "mean_loss": np.mean(losses), "episode_length": env.timer, "height_loss": env.height_lost,
                       "avg_dist_from_skid": np.mean(env.distance_from_checkpoint_per_step),
                       "reward_counts": env.reward_counters,
                       "learning_rate": learning_rate, "last_checkpoint": env.checkpoint})

        print('episode:', i, 'steps:', total_steps, 'score: %.2f' % accumulated_reward,
              'avg score: %.2f' % avg_score, "dist: %.2f" % env.distance_traveled,
              "loss: %.3f" % np.mean(losses), 'learn_rate: %.7f' % learning_rate,
              'eps: %.2f' % agent.epsilon if agent.epsilon > agent.eps_min else '')


# Update graph.html to show the reward counters
def update_graph_html(wandb_url):
    import os
    import json

    if not os.path.exists("graph.html"):
        return

    with open("graph.html", "r") as f:
        html = f.read()

    # Find the iframe and replace the src with the wandb URL
    iframe_start = html.find("<iframe")
    iframe_end = html.find("</iframe>")
    iframe = html[iframe_start:iframe_end]

    src_start = iframe.find("src=")
    src_end = iframe.find(" ", src_start)
    src = iframe[src_start:src_end]

    src = src.replace("src=", "").replace('"', "").replace("'", "")
    html = html.replace(src, f'{wandb_url}')

    with open("graph.html", "w") as f:
        f.write(html)


if __name__ == '__main__':
    start_training()
