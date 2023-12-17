import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import namedtuple

import wandb

import sys


enable_wandb = "pydevd" not in sys.modules


class DeepQNetwork(nn.Module):
    def __init__(self, lr, feature_count, hidden_dims, n_actions, num_layers=3, lstm_units=256):
        super(DeepQNetwork, self).__init__()

        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.num_layers = num_layers
        self.hidden_dims_halved = int(hidden_dims/2)
        self.lstm_units = lstm_units

        self.fc0 = nn.Linear(feature_count-10, self.hidden_dims_halved)
        # self.bn0 = nn.BatchNorm1d(self.hidden_dims)

        self.raycast_cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Flatten(),
            nn.Linear(32 * 3, self.hidden_dims_halved)  # Adjust the input features of nn.Linear
        )

        self.lstm = nn.LSTM(self.hidden_dims, lstm_units, num_layers, batch_first=True)

        self.fc1 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn1 = nn.BatchNorm1d(self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn2 = nn.BatchNorm1d(self.hidden_dims)

        self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn3 = nn.BatchNorm1d(self.hidden_dims)

        # Output layer
        self.value_stream = nn.Linear(self.hidden_dims, 1)
        self.advantage_stream = nn.Linear(self.hidden_dims, self.n_actions)

        # self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-6)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'max',
            factor=0.95,
            patience=300,
            threshold=0.5,
            min_lr=1e-6,
            threshold_mode='abs'
        )

        # self.scheduler = T.optim.lr_scheduler.CyclicLR(
        #     self.optimizer,
        #     base_lr=lr,
        #     max_lr=lr*10,
        #     step_size_up=7500,
        #     step_size_down=20000,
        #     cycle_momentum=False,
        #     mode='exp_range',
        #     gamma=1.0
        # )

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, hidden_state=None, cell_state=None):
        if hidden_state is None or cell_state is None:
            hidden_state = T.zeros(self.num_layers, state.size(0), self.lstm_units).to(self.device)
            cell_state = T.zeros(self.num_layers, state.size(0), self.lstm_units).to(self.device)

        x = F.leaky_relu(self.fc0(state[:, :, :13]), 0.01)
        # x = self.bn0(x)

        # Parts of the state gets pre-processed by a CNN

        # Raycast states are the last 10 features of each observation
        batch_size, num_observations, feature_size = state.shape

        # Separate the raycasting data
        raycasting_data = state[:, :, -10:]  # Last 10 features are raycasting data

        # Reshape for batch processing:
        # New shape: [batch_size * num_observations, 2, 5]
        raycasting_data = raycasting_data.reshape(-1, 2, 5)

        # Process all raycasting data through the CNN in one go
        cnn_out = F.leaky_relu(self.raycast_cnn(raycasting_data), 0.01)

        # Reshape the CNN output back to [batch_size, num_observations, new_feature_size]
        cnn_out = cnn_out.reshape(batch_size, num_observations, -1)

        # Concatenate the CNN output with the non-raycasting part of state
        x = T.cat((x, cnn_out), dim=2)

        # LSTM
        out, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))

        x = F.leaky_relu(self.fc1(out[:, -1, :]), 0.01)
        x = self.bn1(x)

        x = F.leaky_relu(self.fc2(x), 0.01)
        x = self.bn2(x)

        x = F.leaky_relu(self.fc3(x), 0.01)
        x = self.bn3(x)

        value = self.value_stream(x)
        advantages = self.advantage_stream(x)

        actions = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return actions, (hidden_state, cell_state)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


Transition = namedtuple('Transition', (
'state', 'action', 'reward', 'next_state', 'done', 'hidden_state', 'cell_state', 'next_hidden_state',
'next_cell_state'))



class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def add(self, state, action, reward, next_state, done, next_hidden_state, next_cell_state):
        max_priority = max(self.priorities, default=1.0)

        state = T.tensor(state, dtype=T.float32).to(self.device)
        action = T.tensor(action, dtype=T.int64).to(self.device)
        reward = T.tensor(reward, dtype=T.float32).to(self.device)
        next_state = T.tensor(next_state, dtype=T.float32).to(self.device)
        done = T.tensor(done, dtype=T.bool).to(self.device)

        next_hidden_state = next_hidden_state.detach().clone().to(self.device)
        next_cell_state = next_cell_state.detach().clone().to(self.device)

        # Link "current" hidden state to previous transitions' next_hidden_state
        if len(self.buffer) > 0:
            hidden_state = self.buffer[self.position-1][7]
            cell_state = self.buffer[self.position-1][8]
        else:
            hidden_state = next_hidden_state.detach().clone().to(self.device)
            cell_state = next_cell_state.detach().clone().to(self.device)

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, hidden_state, cell_state, next_hidden_state, next_cell_state))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, hidden_state, cell_state, next_hidden_state, next_cell_state)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if not self.buffer:
            return [], [], [], [], []

        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, replace=True, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = Transition(*zip(*samples))
        states, actions, rewards, next_states, dones = map(lambda x: T.stack(x).to(self.device), batch[:-4])

        return states, actions, rewards, next_states, dones, indices, weights, *map(lambda x: T.stack(x).squeeze(2).permute(1, 0, 2).contiguous(), batch[-4:])

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=500000, eps_end=0.005, eps_dec=9e-5, sequence_length=5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.learn_length = 10

        self.action_space = [i for i in range(n_actions)]
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr=lr, feature_count=input_dims, hidden_dims=256, n_actions=n_actions)
        self.Q_target = DeepQNetwork(lr=lr, feature_count=input_dims, hidden_dims=256, n_actions=n_actions)
        self.Q_target.freeze()
        self.update_target_network()

        self.state_memory = np.zeros((self.mem_size, self.sequence_length, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.sequence_length, input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.keyframe_size = [0] * self.mem_size
        self.hidden_state_memory = [None] * self.mem_size
        self.cell_state_memory = [None] * self.mem_size

        self.hidden_state = None
        self.cell_state = None

        self.replay_buffer = PrioritizedReplayBuffer(self.mem_size)

    def start_new_episode(self):
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
            self.hidden_state = None

        if self.cell_state is not None:
            self.cell_state = self.cell_state.detach()
            self.cell_state = None

        self.hidden_state = T.zeros(self.Q_eval.num_layers, 1, self.Q_eval.lstm_units).to(self.Q_eval.device)
        self.cell_state = T.zeros(self.Q_eval.num_layers, 1, self.Q_eval.lstm_units).to(self.Q_eval.device)

    def update_target_network(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def store_transition(self, state_sequence, action, reward, next_state_sequence, done):
        # FIXME: Storing the hidden and cell states for each state is wasteful.
        self.replay_buffer.add(state_sequence, action, reward, next_state_sequence, done,
                               self.hidden_state,
                               self.cell_state
                               )

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

            actions = actions[0]
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, num_batches=1, terminal_learn=False, average_reward=0.0):
        batch_size = self.batch_size * num_batches

        if len(self.replay_buffer.buffer) < batch_size:
            return 0

        if terminal_learn:
            old_state_dict = {name: param.clone() for name, param in self.Q_eval.named_parameters()}

        self.Q_eval.train()
        self.Q_eval.optimizer.zero_grad()

        (states, actions, rewards, next_states, dones, indices, weights,
         hidden_states, cell_states, next_hidden_states, next_cell_states) = self.replay_buffer.sample(batch_size, beta=0.4)

        # Forward pass for current and next state batches
        _actions, _ = self.Q_eval(states, hidden_state=hidden_states, cell_state=cell_states)
        q_eval = _actions.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        q_next, _ = self.Q_eval(next_states, hidden_state=next_hidden_states, cell_state=next_cell_states)
        q_target_next, _ = self.Q_target(next_states, hidden_state=next_hidden_states, cell_state=next_cell_states)

        max_next_actions = T.argmax(q_next, dim=1)
        max_q_next = q_target_next.gather(1, max_next_actions.unsqueeze(-1)).squeeze(-1)
        max_q_next[dones] = 0.0

        # Calculate the Q target and loss
        q_target = rewards + self.gamma * max_q_next

        td_error = abs(q_target - q_eval).detach().cpu().numpy()

        self.replay_buffer.update_priorities(indices, td_error + 1e-5)

        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()

        if terminal_learn and enable_wandb:
            for name, param in self.Q_eval.named_parameters():
                if param.grad is not None:
                    wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu().numpy())}, commit=False)

        T.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=0.5)

        self.Q_eval.optimizer.step()

        for _ in range(num_batches):
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                else self.eps_min

        if terminal_learn:
            self.Q_eval.scheduler.step(average_reward)

            if enable_wandb:
                for name, param in self.Q_eval.named_parameters():
                    if name in old_state_dict:
                        diff = (param - old_state_dict[name]).abs().max()
                        wandb.log({f"weight_change/{name}": diff}, commit=False)

        return loss.item()
