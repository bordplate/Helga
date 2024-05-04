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
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, 1)

        self.actor.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.hidden_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)
        self.cell_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)

    def reset_lstm_states(self):
        self.hidden_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)
        self.cell_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)

    def set_action_std(self, new_action_std):
        self.actor.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    # def act(self, state):
    #     action_mean, _, (self.hidden_state, self.cell_state) = self.actor.sample(state, self.hidden_state, self.cell_state)
    #     cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
    #     dist = MultivariateNormal(action_mean, cov_mat)
    #
    #     action = dist.sample()
    #     action_logprob = dist.log_prob(action)
    #     state_val = self.critic(state)
    #
    #     return action.detach(), action_logprob.detach(), state_val.detach()
    #
    def act(self, state):
        action, action_logprob, (self.hidden_state, self.cell_state), mu, log_std =\
            self.actor.sample(state, self.hidden_state, self.cell_state)

        state_value = self.critic(state)

        return action.detach(), action_logprob.detach(), state_value.detach(), mu, log_std

    # def evaluate(self, state, action, hidden_state=None, cell_state=None):
    #     action_mean, (new_hidden_state, new_cell_state) = self.actor(state, hidden_state, cell_state)
    #
    #     action_var = self.action_var.expand_as(action_mean)
    #     cov_mat = torch.diag_embed(action_var).to(device)
    #     dist = MultivariateNormal(action_mean, cov_mat)
    #
    #     if self.action_dim == 1:
    #         action = action.reshape(-1, self.action_dim)
    #
    #     action_logprobs = dist.log_prob(action)
    #     dist_entropy = dist.entropy()
    #     dist_entropy = -torch.mean(dist_entropy)
    #
    #     state_values = self.critic(state)
    #
    #     return action_logprobs, state_values, dist_entropy, new_hidden_state, new_cell_state

    # def evaluate(self, state, action, hidden_state=None, cell_state=None, old_mus=None, old_log_stds=None):
    #     action_mean, action_log_std, (new_hidden_state, new_cell_state) = self.actor(state, hidden_state, cell_state)
    #     action_std = action_log_std.exp()
    #
    #     normal = Normal(action_mean, action_std)
    #
    #     action_logprobs = normal.log_prob(action)
    #     action_logprobs = action_logprobs.sum(1, keepdim=True)  # Sum log probs across dimensions
    #     #action_logprobs -= torch.log(1 - action.pow(2) + 1e-6).sum(1, keepdim=True)  # Adjustment for tanh
    #
    #     dist_entropy = normal.entropy().mean()  # .mean()
    #     state_value = self.critic(state)
    #
    #     # Calculate kl divergence
    #     old_normal = Normal(old_mus, old_log_stds.exp())
    #     kl_div = torch.distributions.kl.kl_divergence(old_normal, normal).mean()
    #
    #     return action_logprobs, state_value, dist_entropy, new_hidden_state, new_cell_state, kl_div

    def evaluate(self, state, action, hidden_state=None, cell_state=None, old_mus=None, old_log_stds=None):
        mu, log_std, (new_hidden_state, new_cell_state) = self.actor(state, hidden_state, cell_state)
        std = log_std.exp()
        dist = Normal(mu, std)

        # Transform actions to the original action space before calculating log probs
        # Assuming action input is already in the transformed [-max_action, max_action] range
        tanh_action = action / self.actor.max_action
        # Calculate the original actions from tanh actions
        # Inverse tanh can be computed as artanh(x), but let's directly use actions for log_prob
        action_logprobs = dist.log_prob(tanh_action).sum(axis=-1, keepdim=True)
        action_logprobs -= torch.log(self.actor.max_action * (1 - tanh_action.pow(2)) + 1e-6).sum(axis=1, keepdim=True)
        dist_entropy = dist.entropy().mean()

        # KL Divergence if old_mus and old_log_stds are provided
        if old_mus is not None and old_log_stds is not None:
            old_std = old_log_stds.exp().squeeze()
            old_dist = Normal(old_mus.squeeze(), old_std)
            kl_div = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
        else:
            kl_div = None

        state_value = self.critic(state)

        return action_logprobs, state_value, dist_entropy, new_hidden_state, new_cell_state, kl_div


class Actor(nn.Module):
    def __init__(self, feature_count, action_dim):
        super(Actor, self).__init__()

        self.action_dim = action_dim

        self.hidden_dims = 256
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        self.num_layers = 2
        self.max_action = 1.0

        # self.bn0 = nn.BatchNorm1d(feature_count * 8)
        # self.norm = nn.LayerNorm([8, feature_count])
        # self.instance_norm = nn.InstanceNorm1d(143)

        self.fc0 = nn.Linear(feature_count - 128, self.hidden_dims_halved)
        # self.bn0 = nn.BatchNorm1d(self.hidden_dims)

        self.raycast_cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2, 2),
            # PrintSize(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            # PrintSize(),
            nn.Flatten(),
            # PrintSize(),
            nn.Linear(32 * 4 * 4, self.hidden_dims_halved)  # Adjust the input features of nn.Linear
        )

        self.lstm = nn.LSTM(self.hidden_dims, self.hidden_dims, self.num_layers, batch_first=True)

        self.fc1 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn1 = nn.BatchNorm1d(self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn2 = nn.BatchNorm1d(self.hidden_dims)

        self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn3 = nn.BatchNorm1d(self.hidden_dims)

        self.fc4 = nn.Linear(self.hidden_dims, self.action_dim)

        self.log_std = nn.Linear(self.hidden_dims, self.action_dim)

    # Initialize the weights and biases of the model
    def init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.zeros_(layer.bias)

    def forward(self, state, hidden_state=None, cell_state=None):
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(self.num_layers, state.size(0), self.hidden_dims).to(device)
            cell_state = torch.zeros(self.num_layers, state.size(0), self.hidden_dims).to(device)

        # state = self.bn0(state.reshape(state.shape[0], -1)).reshape(state.shape[0], 8, -1)
        # state = self.norm(state)
        # state = self.instance_norm(state.permute(0, 2, 1)).permute(0, 2, 1)

        x = F.leaky_relu(self.fc0(state[:, :, :15]), 0.01)
        # x = self.bn0(x.reshape(state.shape[0], -1)).reshape(state.shape[0], 8, self.hidden_dims_halved)

        # Parts of the state gets pre-processed by a CNN

        # Raycast states are the last 10 features of each observation
        batch_size, num_observations, feature_size = state.shape

        # Separate the raycasting data
        raycasting_data = state[:, :, -128:]  # Last 128 features are raycasting data

        distance_data = raycasting_data[:, :, :64]
        classes_data = raycasting_data[:, :, 64:]

        distance_data = distance_data.reshape(-1, 8, 8)
        classes_data = classes_data.reshape(-1, 8, 8)

        # Combine into a single tensor with 2 channels
        raycasting_data = torch.stack((distance_data, classes_data), dim=1)
        # raycasting_data = T.tensor(raycasting_data, dtype=T.float).unsqueeze(1).to(self.device)

        # Process all raycasting data through the CNN in one go
        cnn_out = F.leaky_relu(self.raycast_cnn(raycasting_data), 0.01)

        # Reshape the CNN output back to [batch_size, num_observations, new_feature_size]
        cnn_out = cnn_out.reshape(batch_size, num_observations, -1)

        # Concatenate the CNN output with the non-raycasting part of state
        x = torch.cat((x, cnn_out), dim=2)

        # LSTM
        out, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))

        x = F.leaky_relu(self.fc1(out[:, -1, :]), 0.01)
        # x = self.bn1(x)

        x = F.leaky_relu(self.fc2(x), 0.01)
        # x = self.bn2(x)

        x = F.leaky_relu(self.fc3(x), 0.01)
        # x = self.bn3(x)

        mu = self.fc4(x)

        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)

        return mu, log_std, (hidden_state, cell_state)

    # def sample(self, state, hidden_state=None, cell_state=None):
    #     mu, log_std, states = self.forward(state, hidden_state, cell_state)
    #
    #     # Use provided action_var to adjust the std
    #     std = log_std.exp()
    #     dist = Normal(mu, std)
    #     action = dist.sample()  # Sample an action
    #     log_prob = dist.log_prob(action)  # Calculate the log probability of the sampled action
    #     log_prob = log_prob.sum(1, keepdim=True)  # Sum log probs across dimensions if needed
    #
    #     tanh_action = torch.tanh(action) * self.max_action
    #     log_prob -= torch.log(self.max_action * (1 - tanh_action.pow(2)) + 1e-6).sum(axis=1, keepdim=True)
    #
    #     return tanh_action, log_prob, states, mu, log_std

    def sample(self, state, hidden_state=None, cell_state=None):
        mu, log_std, (hidden_state, cell_state) = self(state, hidden_state, cell_state)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)

        # Apply tanh to action samples to ensure they are within the desired range and adjust log_prob accordingly
        tanh_action = torch.tanh(action) * self.max_action
        log_prob -= torch.log(self.max_action * (1 - tanh_action.pow(2)) + 1e-6).sum(axis=1, keepdim=True)

        return tanh_action, log_prob, (hidden_state, cell_state), mu, log_std

    def logprob(self, state, action, hidden_state=None, cell_state=None):
        obs = np.array([state])
        state_sequence = torch.tensor(obs, dtype=torch.float).to(device)
        action = action.to(device)

        mu, log_std, _ = self(state_sequence, hidden_state, cell_state)
        std = log_std.exp()
        dist = Normal(mu, std)

        # Calculate the log probability of the action
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        log_prob -= torch.log(self.max_action * (1 - mu.pow(2)) + 1e-6).sum(axis=1, keepdim=True)

        return log_prob

class Critic(nn.Module):
    def __init__(self, feature_count, action_dim):
        super(Critic, self).__init__()

        self.action_dim = action_dim

        self.hidden_dims = 256
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        # self.norm = nn.LayerNorm([8, feature_count])

        self.fc0 = nn.Linear(feature_count - 128, self.hidden_dims_halved)
        # self.bn0 = nn.BatchNorm1d(self.hidden_dims)

        self.raycast_cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2, 2),
            # PrintSize(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            # PrintSize(),
            nn.Flatten(),
            # PrintSize(),
            nn.Linear(32 * 4 * 4, self.hidden_dims_halved)  # Adjust the input features of nn.Linear
        )

        self.fc1 = nn.Linear(self.hidden_dims, self.hidden_dims)
        # self.bn1 = nn.BatchNorm1d(self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        # self.bn2 = nn.BatchNorm1d(self.hidden_dims)

        self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims)
        # self.bn3 = nn.BatchNorm1d(self.hidden_dims)

        self.fc4 = nn.Linear(self.hidden_dims, self.action_dim)

    def forward(self, state):
        # state = self.norm(state)

        x = F.leaky_relu(self.fc0(state[:, :, :15]), 0.01)
        # x = self.bn0(x)

        # Parts of the state gets pre-processed by a CNN

        # Raycast states are the last 10 features of each observation
        batch_size, num_observations, feature_size = state.shape

        # Separate the raycasting data
        raycasting_data = state[:, :, -128:]  # Last 128 features are raycasting data

        distance_data = raycasting_data[:, :, :64]
        classes_data = raycasting_data[:, :, 64:]

        distance_data = distance_data.reshape(-1, 8, 8)
        classes_data = classes_data.reshape(-1, 8, 8)

        # Combine into a single tensor with 2 channels
        raycasting_data = torch.stack((distance_data, classes_data), dim=1)

        # Process all raycasting data through the CNN in one go
        cnn_out = F.leaky_relu(self.raycast_cnn(raycasting_data), 0.01)

        # Reshape the CNN output back to [batch_size, num_observations, new_feature_size]
        cnn_out = cnn_out.reshape(batch_size, num_observations, -1)

        # Concatenate the CNN output with the non-raycasting part of state
        x = torch.cat((x, cnn_out), dim=2)

        x = F.leaky_relu(self.fc1(x[:, -1, :]), 0.01)
        # x = self.bn1(x)

        x = F.leaky_relu(self.fc2(x), 0.01)
        # x = self.bn2(x)

        x = F.leaky_relu(self.fc3(x), 0.01)
        # x = self.bn3(x)

        x = self.fc4(x)

        return x
