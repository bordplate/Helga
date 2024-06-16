import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torch.distributions import Categorical
import torch.nn.functional as F

from .PositionalEncoding import PositionalEncoding


import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, log_std, device='cpu'):
        super(ActorCritic, self).__init__()
        self.device = device

        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, log_std, device=device)
        self.critic = Critic(state_dim, 1, device=device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask=None):
        action, probs, mu, log_std = \
            self.actor.get_action(state, None, mask=mask)

        action_logprob = probs.log_prob(action)

        state_value = self.critic(state)

        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action, mask=None):
        _, probs, action_mean, _ = \
            self.actor.get_action(state, action, mask=mask)

        logprobs = probs.log_prob(action)
        dist_entropy = probs.entropy()

        state_values = self.critic(state)

        return logprobs, state_values, dist_entropy


class Actor(nn.Module):
    def __init__(self, feature_count, action_dim, log_std=-3, max_log_std=0.45, device='cpu'):
        super(Actor, self).__init__()

        self.device = device

        self.action_dim = action_dim
        self.max_log_std = max_log_std
        self.feature_count = feature_count

        self.hidden_dims = 256
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        self.num_layers = 2
        self.max_action = 1.0

        # Linear layer to decode the static features of the state
        self.fc0 = nn.Linear(feature_count - 128, self.hidden_dims_halved)

        # Raycasting data is processed by a separate sequence of linear layers
        self.raycast = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.hidden_dims_halved),
            nn.LeakyReLU()
        )

        self.encoder = nn.Linear(self.hidden_dims, self.hidden_dims)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_dims, nhead=4, dim_feedforward=512,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(self.hidden_dims, self.action_dim)

        self.log_std = nn.Linear(self.hidden_dims, self.action_dim)

        self.init_weights()

    def init_weights(self):
        # Init last layer with very small weights
        self.decoder.weight.data.uniform_(-0.01, 0.01)
        self.decoder.bias.data.zero_()

    def forward(self, state, mask=None):
        """
        Input shape: (batch_size, sequence_length, feature_count)
        """
        # Encode the static features of the state
        x = F.leaky_relu(self.fc0(state[:, :, :self.feature_count - 128]), 0.01)

        # The raycasting parts of the state gets pre-processed by a separate sequence of linear layers
        raycasting_data = state[:, :, -128:]  # Last 128 features are raycasting data

        # Process all raycasting data through the raycasting network
        raycast_out = self.raycast(raycasting_data)

        # Concatenate the raycast output with the non-raycasting part of state
        x = torch.cat((x, raycast_out), dim=2)

        # Transformer
        x = self.encoder(x)
        x = self.transformer_encoder(x)

        x = x[:, -1, :]

        mu = F.tanh(self.decoder(x))

        log_std = F.softplus(self.log_std(x))
        log_std = torch.clamp(log_std, -20, self.max_log_std)

        if mask is not None:
            mu = mu * mask
            log_std = log_std * mask + 1e-5

        return mu, log_std

    def get_action(self, state, action=None, mask=None):
        # If mask is not tensor, spit warning once and convert it to tensor
        if mask is not None and not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

            if self._warned_mask_is_not_tensor is None:
                self._warned_mask_is_not_tensor = True
                print("Warning: Mask is not a tensor, converting it to tensor")

        action_mean, log_std = self(state, mask=mask)
        action_std = log_std

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs, action_mean, log_std


class Critic(nn.Module):
    def __init__(self, feature_count, action_dim, device='cpu'):
        super(Critic, self).__init__()
        self.device = device

        self.action_dim = action_dim
        self.feature_count = feature_count

        self.hidden_dims = 256
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        # Linear layer to decode the static features of the state
        self.fc0 = nn.Linear(feature_count - 128, self.hidden_dims_halved)

        # Raycasting data is processed by a separate sequence of linear layers
        self.raycast = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.hidden_dims_halved),
            nn.LeakyReLU()
        )

        self.encoder = nn.Linear(self.hidden_dims, self.hidden_dims)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_dims, nhead=4, dim_feedforward=768,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.decoder = nn.Linear(self.hidden_dims, 1)

    def forward(self, state):
        """
        Input shape: (batch_size, sequence_length, feature_count)
        """
        # Encode the static features of the state
        x = F.leaky_relu(self.fc0(state[:, :, :self.feature_count - 128]), 0.01)

        # The raycasting parts of the state gets pre-processed by a separate sequence of linear layers
        raycasting_data = state[:, :, -128:]  # Last 128 features are raycasting data

        # Process all raycasting data through the raycasting network
        raycast_out = self.raycast(raycasting_data)

        # Concatenate the raycast output with the non-raycasting part of state
        x = torch.cat((x, raycast_out), dim=2)

        # Transformer
        x = self.encoder(x)
        x = self.transformer_encoder(x)

        x = x[:, -1, :]

        return self.decoder(x)
