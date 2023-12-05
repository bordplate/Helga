import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import wandb

import sys


enable_wandb = "pydevd" not in sys.modules


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


class Attention(nn.Module):
    def __init__(self, hidden_dims):
        super(Attention, self).__init__()
        # Use a linear layer to transform the LSTM output
        self.linear = nn.Linear(hidden_dims, hidden_dims)
        # Another linear layer to compute the attention scores
        self.attention_score = nn.Linear(hidden_dims, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dims)
        # Transform lstm_output
        transformed_output = T.tanh(self.linear(lstm_output))
        # Compute attention scores for each time step
        attention_scores = self.attention_score(transformed_output) # Shape: (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1) # Shape: (batch, seq_len, 1)

        # Compute context vector as the weighted sum of LSTM outputs
        context_vector = T.sum(lstm_output * attention_weights, dim=1) # Shape: (batch, hidden_dims)
        return context_vector, attention_weights.squeeze(-1)


class DeepQNetwork(nn.Module):
    def __init__(self, lr, feature_count, hidden_dims, n_actions, num_layers=2, lstm_units=64):
        super(DeepQNetwork, self).__init__()

        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.num_layers = num_layers
        self.hidden_dims_halved = int(hidden_dims/2)
        self.lstm_units = lstm_units

        #T.autograd.set_detect_anomaly(True)

        self.raycast_cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=2, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2)
        )

        # Assuming feature_count is the number of features in each timestep
        # self.lstm = nn.LSTM(feature_count, lstm_units, num_layers, batch_first=True)
        self.lstm = nn.LSTM(feature_count, lstm_units, num_layers, batch_first=True)
        # self.attention = Attention(self.hidden_dims_halved,)
        self.attention = MultiHeadAttention(lstm_units, 8)

        self.fc1 = nn.Linear(lstm_units, self.hidden_dims)
        self.bn1 = nn.BatchNorm1d(self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn2 = nn.BatchNorm1d(self.hidden_dims)

        self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims_halved)
        self.bn3 = nn.BatchNorm1d(self.hidden_dims_halved)

        # Output layer
        self.fc4 = nn.Linear(self.hidden_dims_halved, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        #self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)

        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'max',
            factor=0.9,
            patience=150,
            threshold=0.1,
            threshold_mode='abs'
        )

        # self.loss = nn.MSELoss()
        self.loss = nn.HuberLoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, attention=False, hidden_state=None, cell_state=None):
        if hidden_state is None or cell_state is None:
            hidden_state = T.zeros(self.num_layers, state.size(0), self.lstm_units).to(self.device)
            cell_state = T.zeros(self.num_layers, state.size(0), self.lstm_units).to(self.device)

        # Shape of state: [batch_size, num_observations, feature_size]

        # Parts of the state gets pre-processed by a CNN

        # Raycast states are the last 10 features of each observation
        batch_size, num_observations, feature_size = state.shape

        # Separate the raycasting data
        raycasting_data = state[:, :, -10:]  # Last 10 features are raycasting data

        # Reshape for batch processing:
        # New shape: [batch_size * num_observations, 2, 5]
        raycasting_data = raycasting_data.reshape(-1, 2, 5)

        # Process all raycasting data through the CNN in one go
        cnn_out = self.raycast_cnn(raycasting_data)

        # Reshape the CNN output back to [batch_size, num_observations, new_feature_size]
        cnn_out = cnn_out.reshape(batch_size, num_observations, -1)

        # Concatenate the CNN output with the non-raycasting part of state
        state_without_raycasting = state[:, :, :-10]
        state = T.cat((state_without_raycasting, cnn_out), dim=2)

        # LSTM
        out, (hidden_state, cell_state) = self.lstm(state, (hidden_state, cell_state))
        # context_vector, attention_weights = self.attention(out, out, out)
        context_vector, attention_weights = self.attention(out)

        x = F.leaky_relu(self.fc1(context_vector[:, -1, :]), 0.01)
        # x = F.relu(self.fc1(out[:, -1, :]))
        x = self.bn1(x)

        x = F.leaky_relu(self.fc2(x), 0.01)
        x = self.bn2(x)

        x = F.leaky_relu(self.fc3(x), 0.01)
        x = self.bn3(x)

        actions = self.fc4(x)

        if attention:
            return actions, attention_weights, (hidden_state, cell_state)

        return actions, (hidden_state, cell_state)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.001, eps_dec=6e-4, sequence_length=5):
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

        self.Q_eval = DeepQNetwork(lr=lr, feature_count=input_dims, hidden_dims=512, n_actions=n_actions)
        self.Q_target = DeepQNetwork(lr=lr, feature_count=input_dims, hidden_dims=512, n_actions=n_actions)
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

        self.hidden_state_learn = None
        self.cell_state_learn = None

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
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state_sequence
        self.new_state_memory[index] = next_state_sequence
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        keyframe_index = index - (index % 100)

        # Store LSTM hidden and cell states every keyframe
        if self.mem_cntr % 100 == 0 and self.hidden_state is not None and self.cell_state is not None:
            num_states = np.random.randint(10, 90)
            self.keyframe_size[index] = num_states

            self.hidden_state_memory[index] = self.hidden_state.detach().clone().to(self.Q_eval.device)
            self.cell_state_memory[index] = self.cell_state.detach().clone().to(self.Q_eval.device)
        elif index % 100 <= self.keyframe_size[keyframe_index]:
            # Make sure we don't cross episode boundaries
            if done:
                self.keyframe_size[keyframe_index] = index - keyframe_index

            self.hidden_state_memory[index] = self.hidden_state.detach().clone().to(self.Q_eval.device)
            self.cell_state_memory[index] = self.cell_state.detach().clone().to(self.Q_eval.device)

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

            actions = actions[0]
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, terminal_learn=False, average_reward=0.0):
        if self.mem_cntr < self.batch_size or self.mem_cntr < 400:
            return 0

        if terminal_learn:
            old_state_dict = {name: param.clone() for name, param in self.Q_eval.named_parameters()}

        self.Q_eval.train()
        self.Q_eval.optimizer.zero_grad()
        max_mem = int(min(self.mem_cntr, self.mem_size) / 100)

        if max_mem <= int(self.batch_size / 10) + 1:
            return 0

        probabilities = np.linspace(0.000000000001, 1, max_mem)
        probabilities /= probabilities.sum()

        batch = np.random.choice(max_mem, int(self.batch_size/10), replace=False, p=probabilities)
        processed_states = 0

        # Accumulators for data and states
        state_batches, new_state_batches = [], []
        action_batches, reward_batches, terminal_batches = [], [], []
        hidden_states, cell_states = [], []
        hidden_states_next, cell_states_next = [], []

        for batch_index in batch:
            if processed_states > self.batch_size:
                break

            processed_states += 1

            keyframe_index = batch_index * 100
            num_states = self.keyframe_size[keyframe_index]

            if keyframe_index + num_states > self.mem_cntr or keyframe_index + num_states > self.mem_size or num_states < 5:
                continue

            sequential_batch = np.arange(keyframe_index + 1, keyframe_index + num_states)
            processed_states += num_states

            # Accumulate data
            state_batches.append(T.tensor(self.state_memory[sequential_batch]).to(self.Q_eval.device))
            new_state_batches.append(T.tensor(self.new_state_memory[sequential_batch]).to(self.Q_eval.device))
            action_batches.append(T.tensor(self.action_memory[sequential_batch], dtype=T.long).to(self.Q_eval.device))
            reward_batches.append(T.tensor(self.reward_memory[sequential_batch]).to(self.Q_eval.device))
            terminal_batches.append(T.tensor(self.terminal_memory[sequential_batch]).to(self.Q_eval.device))

            # Accumulate states
            hidden_states.append(T.cat([self.hidden_state_memory[idx - 1] if self.hidden_state_memory[
                                                                                 idx - 1] is not None else T.zeros(
                self.Q_eval.num_layers, 1, self.Q_eval.lstm_units).to(self.Q_eval.device) for idx in sequential_batch], dim=1))
            cell_states.append(T.cat([self.cell_state_memory[idx - 1] if self.cell_state_memory[
                                                                             idx - 1] is not None else T.zeros(
                self.Q_eval.num_layers, 1, self.Q_eval.lstm_units).to(self.Q_eval.device) for idx in sequential_batch], dim=1))

            hidden_states_next.append(T.cat([self.hidden_state_memory[idx] if self.hidden_state_memory[
                                                                                    idx] is not None else T.zeros(
                    self.Q_eval.num_layers, 1, self.Q_eval.lstm_units).to(self.Q_eval.device) for idx in sequential_batch], dim=1))
            cell_states_next.append(T.cat([self.cell_state_memory[idx] if self.cell_state_memory[
                                                                                idx] is not None else T.zeros(
                    self.Q_eval.num_layers, 1, self.Q_eval.lstm_units).to(self.Q_eval.device) for idx in sequential_batch], dim=1))

        if processed_states <= 0:
            print("WARNING: No processed states!")
            return 0

        # Concatenate all accumulated data and states
        state_batch = T.cat(state_batches, dim=0)
        new_state_batch = T.cat(new_state_batches, dim=0)
        action_batch = T.cat(action_batches, dim=0)
        reward_batch = T.cat(reward_batches, dim=0)
        terminal_batch = T.cat(terminal_batches, dim=0)
        hidden_state = T.cat(hidden_states, dim=1)
        cell_state = T.cat(cell_states, dim=1)
        hidden_state_next = T.cat(hidden_states_next, dim=1)
        cell_state_next = T.cat(cell_states_next, dim=1)

        # Forward pass for current and next state batches
        actions, (hidden_state, cell_state) = self.Q_eval(state_batch, hidden_state=hidden_state, cell_state=cell_state)
        q_eval = actions.gather(1, action_batch.unsqueeze(-1)).squeeze(-1)

        q_next, _ = self.Q_eval(new_state_batch, hidden_state=hidden_state_next, cell_state=cell_state_next)
        q_target_next, _ = self.Q_target(new_state_batch, hidden_state=hidden_state_next, cell_state=cell_state_next)

        max_next_actions = T.argmax(q_next, dim=1)
        max_q_next = q_target_next.gather(1, max_next_actions.unsqueeze(-1)).squeeze(-1)
        terminal_batch = terminal_batch.bool()
        max_q_next[terminal_batch] = 0.0

        # Calculate the Q target and loss
        q_target = reward_batch + self.gamma * max_q_next
        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()

        if terminal_learn and enable_wandb:
            for name, param in self.Q_eval.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu().numpy())}, commit=False)
                    wandb.log({f"grad/{name}_norm": grad_norm}, commit=False)

        T.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=0.5)

        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

        if terminal_learn:
            # Don't update the scheduler if we're not learning
            if self.epsilon <= self.eps_min:
                self.Q_eval.scheduler.step(average_reward)

            if enable_wandb:
                for name, param in self.Q_eval.named_parameters():
                    if name in old_state_dict:
                        diff = (param - old_state_dict[name]).abs().max()
                        wandb.log({f"weight_change/{name}": diff}, commit=False)

        return loss.item()
