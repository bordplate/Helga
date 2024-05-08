import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torch.distributions import Categorical
import torch.nn.functional as F

import numpy as np

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, log_std):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, log_std)
        self.critic = Critic(state_dim, 1)

        self.hidden_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)
        self.cell_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)

    def reset_lstm_states(self):
        self.hidden_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)
        self.cell_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action, probs, (self.hidden_state, self.cell_state), mu, log_std =\
            self.actor.get_action(state, None, self.hidden_state, self.cell_state)

        action_logprob = probs.log_prob(action)

        state_value = self.critic(state)

        return action.detach(), action_logprob.detach(), state_value.detach(), mu, log_std

    def evaluate(self, state, action, hidden_state=None, cell_state=None, old_mus=None, old_log_stds=None):
        _, probs, (new_hidden_state, new_cell_state), action_mean, _ = \
            self.actor.get_action(state, action, hidden_state, cell_state)

        logprobs = probs.log_prob(action)
        dist_entropy = probs.entropy()

        state_values = self.critic(state)

        return logprobs, state_values, dist_entropy, new_hidden_state, new_cell_state, 0


class Actor(nn.Module):
    def __init__(self, feature_count, action_dim, log_std=-3):
        super(Actor, self).__init__()

        self.action_dim = action_dim

        self.hidden_dims = 256
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        self.num_layers = 2
        self.max_action = 1.0

        self.fc0 = nn.Linear(feature_count - 128, self.hidden_dims_halved)

        self.raycast_cnn = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.hidden_dims_halved),
        )

        self.lstm = nn.LSTM(self.hidden_dims, self.hidden_dims, self.num_layers, batch_first=True)

        self.fc1 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc4 = nn.Linear(self.hidden_dims, self.action_dim)

        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

        self.init_weights(log_std=log_std)

    # Initialize the weights and biases of the model
    def init_weights(self, std=np.sqrt(2), bias=0.01, log_std=-3):
        # Orthogonal initialization of the weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, std)
            elif 'bias' in name:
                nn.init.constant_(param, bias)
            elif 'log_std' in name:
                nn.init.constant_(param, log_std)

    def forward(self, state, hidden_state=None, cell_state=None):
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(self.num_layers, state.size(0), self.hidden_dims).to(device)
            cell_state = torch.zeros(self.num_layers, state.size(0), self.hidden_dims).to(device)

        x = F.leaky_relu(self.fc0(state[:, :, :18]), 0.01)

        # Parts of the state gets pre-processed by a CNN

        # Raycast states are the last 10 features of each observation
        batch_size, num_observations, feature_size = state.shape

        # Separate the raycasting data
        raycasting_data = state[:, :, -128:]  # Last 128 features are raycasting data

        # Process all raycasting data through the CNN in one go
        cnn_out = F.leaky_relu(self.raycast_cnn(raycasting_data), 0.01)

        # Reshape the CNN output back to [batch_size, num_observations, new_feature_size]
        cnn_out = cnn_out.reshape(batch_size, num_observations, -1)

        # Concatenate the CNN output with the non-raycasting part of state
        x = torch.cat((x, cnn_out), dim=2)

        # LSTM
        out, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))

        x = F.leaky_relu(self.fc1(out[:, -1, :]), 0.01)

        x = F.leaky_relu(self.fc2(x), 0.01)

        mu = F.tanh(self.fc4(x))

        log_std = F.softplus(self.log_std)
        log_std = torch.clamp(log_std, -20, 2)

        return mu, log_std, (hidden_state, cell_state)

    def get_action(self, state, action=None, hidden_state=None, cell_state=None):
        action_mean, log_std, (hidden_state, cell_state) = self(state, hidden_state, cell_state)
        action_std = log_std

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs, (hidden_state, cell_state), action_mean, log_std

class Critic(nn.Module):
    def __init__(self, feature_count, action_dim):
        super(Critic, self).__init__()

        self.action_dim = action_dim

        self.hidden_dims = 128
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        self.fc0 = nn.Linear(feature_count - 128, self.hidden_dims_halved)

        self.raycast_cnn = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.hidden_dims_halved),
        )

        self.fc1 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc4 = nn.Linear(self.hidden_dims, self.action_dim)

        self.init_weights()

    def init_weights(self, std=1, bias=0.01):
        # Orthogonal initialization of the weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, std)
            elif 'bias' in name:
                nn.init.constant_(param, bias)

    def forward(self, state):
        x = F.leaky_relu(self.fc0(state[:, :, :18]), 0.01)

        # Parts of the state gets pre-processed by a CNN

        # Raycast states are the last 10 features of each observation
        batch_size, num_observations, feature_size = state.shape

        # Separate the raycasting data
        raycasting_data = state[:, :, -128:]  # Last 128 features are raycasting data

        # Process all raycasting data through the CNN in one go
        cnn_out = F.leaky_relu(self.raycast_cnn(raycasting_data), 0.01)

        # Reshape the CNN output back to [batch_size, num_observations, new_feature_size]
        cnn_out = cnn_out.reshape(batch_size, num_observations, -1)

        # Concatenate the CNN output with the non-raycasting part of state
        x = torch.cat((x, cnn_out), dim=2)

        x = F.leaky_relu(self.fc1(x[:, -1, :]), 0.01)

        x = F.leaky_relu(self.fc2(x), 0.01)

        x = self.fc4(x)

        return x
