# import torch as T
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
#
#
# class DeepQNetwork(nn.Module):
#     def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
#         super(DeepQNetwork, self).__init__()
#
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = 512
#         self.fc3_dims = fc2_dims
#         self.n_actions = n_actions
#
#         self.lstm = nn.LSTM(*self.input_dims, fc1_dims, 2, batch_first=True)
#         #self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
#         #self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         #self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
#         self.fc4 = nn.Linear(self.fc1_dims, self.n_actions)
#
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#         self.loss = nn.MSELoss()
#
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)
#
#     def forward(self, state):
#         # x = F.relu(self.fc1(state))
#         # x = F.relu(self.fc2(x))
#         # x = F.relu(self.fc3(x))
#         # actions = self.fc4(x)
#
#         h0 = T.zeros(2, state.size(0), self.fc1_dims).to(self.device)
#         c0 = T.zeros(2, state.size(0), self.fc1_dims).to(self.device)
#
#         out, _ = self.lstm(state, (h0, c0))
#         actions = self.fc4(out[:, -1, :])
#
#         return actions
#
#
# class Agent:
#     def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
#                  max_mem_size=10000000, eps_end=0.001, eps_dec=3e-4):
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.eps_min = eps_end
#         self.eps_dec = eps_dec
#         self.lr = lr
#         self.mem_size = max_mem_size
#         self.batch_size = batch_size
#
#         self.action_space = [i for i in range(n_actions)]
#         self.mem_cntr = 0
#
#         self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
#                                    fc1_dims=256, fc2_dims=256)
#
#         self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
#         self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
#
#         self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
#         self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
#
#     def store_transition(self, state, action, reward, state_, done):
#         index = self.mem_cntr % self.mem_size
#         self.state_memory[index] = state
#         self.new_state_memory[index] = state_
#         self.reward_memory[index] = reward
#         self.action_memory[index] = action
#         self.terminal_memory[index] = done
#
#         self.mem_cntr += 1
#
#     def choose_action(self, observation):
#         if np.random.random() > self.epsilon:
#             obs = np.array(observation)
#             state = T.tensor([obs]).to(self.Q_eval.device)
#             actions = self.Q_eval.forward(state)
#             action = T.argmax(actions).item()
#         else:
#             action = np.random.choice(self.action_space)
#
#         return action
#
#     def learn(self):
#         if self.mem_cntr < self.batch_size:
#             return 0
#
#         self.Q_eval.optimizer.zero_grad()
#
#         max_mem = min(self.mem_cntr, self.mem_size)
#         batch = np.random.choice(max_mem, self.batch_size, replace=False)
#
#         batch_index = np.arange(self.batch_size, dtype=np.int32)
#
#         state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
#         new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
#         reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
#         terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
#
#         action_batch = self.action_memory[batch]
#
#         q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
#         q_next = self.Q_eval.forward(new_state_batch)
#
#         q_next[terminal_batch] = 0.0
#
#         q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
#
#         loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
#         loss.backward()
#         self.Q_eval.optimizer.step()
#
#         self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
#             else self.eps_min
#
#         return loss.item()

I have a test for you. You get 1 point for every correct assessment, and -2 points for every wrong assessment. In the following code there are mistakes that make this reinforcement learning agent for an agent playing a PS2 game significantly weaker. Point out the errors:

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dims, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
        self.head_dim = hidden_dims // num_heads

        assert (
            self.head_dim * num_heads == hidden_dims
        ), "Embedding size must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.query_linear = nn.Linear(hidden_dims, hidden_dims)
        self.key_linear = nn.Linear(hidden_dims, hidden_dims)
        self.value_linear = nn.Linear(hidden_dims, hidden_dims)

        self.fc_out = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, lstm_output):
        batch_size = lstm_output.shape[0]
        query = self.query_linear(lstm_output)
        key = self.key_linear(lstm_output)
        value = self.value_linear(lstm_output)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = T.matmul(query, key.permute(0, 1, 3, 2)) * self.scaling
        attention = T.softmax(energy, dim=-1)

        out = T.matmul(attention, value).permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.hidden_dims)
        out = self.fc_out(out)

        return out, attention


class DeepQNetwork(nn.Module):
    def __init__(self, lr, feature_count, hidden_dims, n_actions, num_layers=2):
        super(DeepQNetwork, self).__init__()

        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.num_layers = num_layers
        self.hidden_dims_halved = int(hidden_dims/2)

        #T.autograd.set_detect_anomaly(True)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=2, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.LayerNorm([2, 5])
        )


        # Assuming feature_count is the number of features in each timestep
        self.lstm = nn.LSTM(feature_count, self.hidden_dims_halved, num_layers, batch_first=True)
        # self.attention = Attention(self.hidden_dims_halved,)
        self.attention = MultiHeadAttention(self.hidden_dims_halved, 8)

        self.fc1 = nn.Linear(self.hidden_dims_halved, self.hidden_dims)
        self.bn1 = nn.BatchNorm1d(self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn2 = nn.BatchNorm1d(self.hidden_dims)

        self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims_halved)
        self.bn3 = nn.BatchNorm1d(self.hidden_dims_halved)

        # Output layer
        self.fc4 = nn.Linear(self.hidden_dims_halved, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-6)
        #self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, attention=False, hidden_state=None, cell_state=None):
        if hidden_state is None or cell_state is None:
            hidden_state = T.zeros(self.num_layers, state.size(0), self.hidden_dims_halved).to(self.device)
            cell_state = T.zeros(self.num_layers, state.size(0), self.hidden_dims_halved).to(self.device)

        # Shape of state: [batch_size, num_observations, feature_size]

        # Parts of the state gets pre-processed by a CNN

        if True:
            batch_size, num_observations, feature_size = state.shape

            # Separate the raycasting data
            raycasting_data = state[:, :, -10:]  # Last 10 features are raycasting data

            # Reshape for batch processing:
            # New shape: [batch_size * num_observations, 2, 5]
            raycasting_data = raycasting_data.reshape(-1, 2, 5)

            # Process all raycasting data through the CNN in one go
            cnn_out = self.cnn(raycasting_data)

            # Reshape the CNN output back to [batch_size, num_observations, new_feature_size]
            cnn_out = cnn_out.reshape(batch_size, num_observations, -1)

            # Concatenate the CNN output with the non-raycasting part of state
            state_without_raycasting = state[:, :, :-10]
            state = T.cat((state_without_raycasting, cnn_out), dim=2)

        # LSTM
        out, (hidden_state, cell_state) = self.lstm(state, (hidden_state, cell_state))
        # context_vector, attention_weights = self.attention(out, out, out)
        context_vector, attention_weights = self.attention(out)

        x = F.relu(self.fc1(context_vector[:, -1, :]))
        # x = F.relu(self.fc1(out[:, -1, :]))
        #x = self.bn1(x)

        x = F.relu(self.fc2(x))
        #x = self.bn2(x)

        x = F.relu(self.fc3(x))
        #x = self.bn3(x)

        actions = self.fc4(x)

        if attention:
            return actions, attention_weights, (hidden_state, cell_state)

        return actions, (hidden_state, cell_state)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=300000, eps_end=0.001, eps_dec=6e-4, sequence_length=5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.action_space = [i for i in range(n_actions)]
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr=lr, feature_count=input_dims, hidden_dims=2048, n_actions=n_actions)
        self.Q_target = DeepQNetwork(lr=lr, feature_count=input_dims, hidden_dims=2048, n_actions=n_actions)
        self.Q_target.freeze()
        self.update_target_network()

        self.state_memory = np.zeros((self.mem_size, self.sequence_length, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.sequence_length, input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.hidden_state = None
        self.cell_state = None

    def start_new_episode(self):
        self.hidden_state = None
        self.cell_state = None

    def update_target_network(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def store_transition(self, state_sequence, action, reward, next_state_sequence, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state_sequence
        self.new_state_memory[index] = next_state_sequence
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation_sequence):
        if np.random.random() > self.epsilon:
            obs = np.array([observation_sequence])
            state_sequence = T.tensor(obs, dtype=T.float).to(self.Q_eval.device)

            self.Q_eval.eval()
            with T.no_grad():
                if self.hidden_state is None or self.cell_state is None:
                    actions, (self.hidden_state, self.cell_state) = self.Q_eval(state_sequence)
                else:
                    actions, (self.hidden_state, self.cell_state) = self.Q_eval(state_sequence, hidden_state=self.hidden_state,
                                                                                cell_state=self.cell_state)

            self.Q_eval.eval()
            actions = self.Q_eval(state_sequence)[0]
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return 0

        self.Q_eval.train()

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        # Create an exponential probability distribution that favors recent experiences
        probabilities = np.linspace(0.000000000001, 1, max_mem)
        probabilities /= probabilities.sum()

        batch = np.random.choice(max_mem, self.batch_size, replace=False, p=probabilities)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)

        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        # Do Double DQN
        # Forward pass for current state batch
        actions, _ = self.Q_eval(state_batch)
        q_eval = actions[batch_index, action_batch]

        # Forward pass for next state batch
        q_next, _ = self.Q_eval(new_state_batch)
        q_target_next, _ = self.Q_target(new_state_batch)

        # Get the action that maximizes Q value for the next states from online network
        max_next_actions = T.argmax(q_next, dim=1)

        # Get the maximum Q value for the next states from target network
        max_q_next = q_target_next[batch_index, max_next_actions]

        # Set the Q value to 0 for all terminal states
        max_q_next[terminal_batch] = 0.0

        # Calculate the Q target
        q_target = reward_batch + self.gamma * max_q_next

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()

        T.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=1.0)

        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
        if self.cell_state is not None:
            self.cell_state = self.cell_state.detach()

        return loss.item()

from agent import Agent
from ratchet_environment import RatchetEnvironment
from watchdog import Watchdog

import time
import numpy as np
import wandb

import os
import random
import torch


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

    import sys

    sequence_length = 30
    enable_wandb = "pydevd" not in sys.modules

    learning_rate = 0.00025
    features = 22
    batch_size = 128
    train_frequency = 4
    target_update_frequency = 150

    learning_rate_schedule = {
        500: 0.00025,
        1000: 0.00025
    }

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

        if i in learning_rate_schedule:
            for g in agent.Q_eval.optimizer.param_groups:
                g['lr'] = learning_rate_schedule[i]

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

            if steps % train_frequency == 0 or done:
                loss = agent.learn()
                losses.append(loss)

            accumulated_reward += reward

            steps += 1
            total_steps += 1

            if total_steps % target_update_frequency == 0:
                agent.update_target_network()

        if env.distance_traveled > furthest_distance:
            furthest_distance = env.distance_traveled
            print(f"New furthest distance: {furthest_distance}")

        scores.append(accumulated_reward)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

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
              "loss: %.3f" % np.mean(losses), 'learning_rate: %.5f' % learning_rate)
