import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # self.actor = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, self.action_dim)
        # )
        # self.critic = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 1)
        # )

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, 1)

        self.hidden_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)
        self.cell_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)

    def reset_lstm_states(self):
        self.hidden_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)
        self.cell_state = torch.zeros(self.actor.num_layers, 1, self.actor.hidden_dims).to(device)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean, (self.hidden_state, self.cell_state) = self.actor(state, self.hidden_state, self.cell_state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action, hidden_state=None, cell_state=None):
        action_mean, (new_hidden_state, new_cell_state) = self.actor(state, hidden_state, cell_state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy, new_hidden_state, new_cell_state


class Actor(nn.Module):
    def __init__(self, feature_count, action_dim):
        super(Actor, self).__init__()

        self.action_dim = action_dim

        self.hidden_dims = 256
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        self.num_layers = 2

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

    def forward(self, state, hidden_state=None, cell_state=None):
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(self.num_layers, state.size(0), self.hidden_dims).to(device)
            cell_state = torch.zeros(self.num_layers, state.size(0), self.hidden_dims).to(device)

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
        x = self.bn1(x)

        x = F.leaky_relu(self.fc2(x), 0.01)
        x = self.bn2(x)

        x = F.leaky_relu(self.fc3(x), 0.01)
        x = self.bn3(x)

        x = torch.tanh(self.fc4(x))

        return x, (hidden_state, cell_state)


class Critic(nn.Module):
    def __init__(self, feature_count, action_dim):
        super(Critic, self).__init__()

        self.action_dim = action_dim

        self.hidden_dims = 256
        self.hidden_dims_halved = int(self.hidden_dims / 2)

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
        self.bn1 = nn.BatchNorm1d(self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn2 = nn.BatchNorm1d(self.hidden_dims)

        self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.bn3 = nn.BatchNorm1d(self.hidden_dims)

        self.fc4 = nn.Linear(self.hidden_dims, self.action_dim)

    def forward(self, state):
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
        x = self.bn1(x)

        x = F.leaky_relu(self.fc2(x), 0.01)
        x = self.bn2(x)

        x = F.leaky_relu(self.fc3(x), 0.01)
        x = self.bn3(x)

        x = self.fc4(x)

        return x
