from agent import Agent
from ratchet_environment import RatchetEnvironment
from watchdog import Watchdog

import time
import numpy as np
import wandb

import os
import random
import torch

import sys


enable_wandb = "pydevd" not in sys.modules

# Add load_model variable at the top of the script
load_model = ""  # Set to model filename to load, otherwise leave as empty string


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
    env = RatchetEnvironment()

    watchdog = Watchdog(env)
    watchdog.start()

    env.open_process()

    random_instance_id = random.randint(0000, 999999)  # Generate a random instance ID

    sequence_length = 30

    learning_rate = 0.0006
    features = 23
    batch_size = 128
    train_frequency = 4
    target_update_frequency = 1000

    steps_since_update = 0
    steps_since_learn = 0

    # learning_rate_schedule = {
    #     500: 1e-3,
    #     1000: 1e-3
    # }

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=batch_size, n_actions=4, eps_end=0.001,
                  input_dims=features, lr=learning_rate, sequence_length=sequence_length)

    if load_model:
        model_path = os.path.join('models_bak', load_model)
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path)
            agent.Q_eval.load_state_dict(checkpoint['model_state_dict'])
            agent.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = checkpoint['epsilon']
            start_episode = checkpoint['episode']

            agent.Q_eval = agent.Q_eval.to(agent.Q_eval.device)

            agent.Q_eval.train()
        else:
            print(f"Model file {model_path} not found. Starting fresh.")

    scores, eps_history = [], []
    n_games = 1000000

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

    print(f"Device: {agent.Q_eval.device}. Instance ID: {random_instance_id}")

    furthest_distance = 0.0

    observation_sequence = np.zeros((sequence_length, features))  # Initialize the sequence

    total_steps = 0

    for i in range(n_games):
        done = False

        # if i in learning_rate_schedule:
        #     for g in agent.Q_eval.optimizer.param_groups:
        #         g['lr'] = learning_rate_schedule[i]

        observation = env.reset()[0]

        # Reinitialize the observation_sequence for the new episode
        observation_sequence = np.zeros((sequence_length, features))  # Reset the sequence
        #new_observation_sequence = np.zeros((sequence_length, features))  # Reset the sequence

        # Set the last observation to the initial observation from the environment
        observation_sequence[-1] = observation

        accumulated_reward = 0

        losses = []

        steps = 0

        if i % 200 == 0 and i != 0:
            # Save the model every 200 episodes
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

        agent.start_new_episode()

        while not done:
            action = agent.choose_action(observation_sequence)
            observation_, reward, done = env.step(action)
            observation = observation_

            # Frame skip to improve temporal resolution
            new_observation_sequence = np.concatenate((observation_sequence[1:], np.array([observation_])), axis=0)
            agent.store_transition(observation_sequence, action, reward, new_observation_sequence, done)

            observation_sequence = new_observation_sequence

            steps_since_update += 1
            steps_since_learn += 1

            accumulated_reward += reward

            steps += 1
            total_steps += 1

            if total_steps % target_update_frequency == 0:
                agent.update_target_network()

        # for i in range(int(steps_since_learn / train_frequency)-1):
        #     loss = agent.learn()
        #     losses.append(loss)

        loss = agent.learn(num_batches=max(1, int(steps_since_learn / train_frequency)), terminal_learn=True, average_reward=accumulated_reward)
        losses.append(loss)

        steps_since_learn = 0

        if steps_since_update > target_update_frequency:
            agent.update_target_network()
            steps_since_update = 0

        avg_score = np.mean(scores[-100:])

        agent.learn(terminal_learn=True, average_reward=avg_score)

        if env.distance_traveled > furthest_distance:
            furthest_distance = env.distance_traveled
            print(f"New furthest distance: {furthest_distance}")

        scores.append(accumulated_reward)
        eps_history.append(agent.epsilon)

        thing_i = 0
        for g in agent.Q_eval.optimizer.param_groups:
            thing_i += 1
            if thing_i > 1:
                print(f"WARNING: More than one optimizer param group found. This is probably a bug.")
                break

            learning_rate = g['lr']

        if enable_wandb:
            wandb.log({"reward": accumulated_reward, "epsilon": agent.epsilon, "average_reward": avg_score,
                       "distance_traveled": env.distance_traveled, "furthest_distance": furthest_distance,
                       "mean_loss": np.mean(losses), "episode_length": env.timer, "height_loss": env.height_lost,
                       "avg_dist_from_skid": np.mean(env.distance_from_skid_per_step),
                       "reward_counts": env.reward_counters, "skid_checkpoints": len(env.skid_checkpoints),
                       "learning_rate": learning_rate, "last_checkpoint": env.checkpoint})

        print('episode:', i, 'steps:', total_steps, 'score: %.2f' % (accumulated_reward),
              'avg score: %.2f' % avg_score, "dist: %.2f" % env.distance_traveled,
              "loss: %.3f" % np.mean(losses), 'learn_rate: %.5f' % learning_rate,
              'eps: %.2f' % agent.epsilon if agent.epsilon > agent.eps_min else '')
