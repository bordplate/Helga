import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class RayycastModule(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(RayycastModule, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.fc0 = nn.Linear(input_dims, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, self.output_dims)

    def size(self):
        return self.output_dims

    def forward(self, x):
        x = F.leaky_relu(self.fc0(x), 0.01)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)

        return x


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, log_std, device='cpu'):
        super(ActorCritic, self).__init__()
        self.device = device

        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, log_std, device=device)
        self.critic = Critic(state_dim, 1, device=device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask=None, hidden_state=None, cell_state=None):
        # if cell_state is None or hidden_state is None:
        #     hidden_state = self.actor.hidden_state
        #     cell_state = self.actor.cell_state

        action, probs, mu, log_std, (hidden_states, cell_states) = \
            self.actor.get_action(state, None, mask=mask, hidden_state=hidden_state, cell_state=cell_state)

        action_logprob = probs.log_prob(action)

        state_value = self.critic(state, hidden_states=hidden_states, cell_states=cell_states)

        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action, mask=None, hidden_state=None, cell_state=None):
        # if cell_state is None or hidden_state is None:
        #     hidden_state = self.actor.hidden_state
        #     cell_state = self.actor.cell_state

        _, probs, action_mean, _, (hidden_states, cell_states) = \
            self.actor.get_action(state, action, mask=mask, hidden_state=hidden_state, cell_state=cell_state)

        logprobs = probs.log_prob(action)
        dist_entropy = probs.entropy()

        state_values = self.critic(state, hidden_states=hidden_states, cell_states=cell_states)

        return logprobs, state_values, dist_entropy

    def start_new_episode(self):
        self.actor.reset_lstm()


class Actor(nn.Module):
    def __init__(self, feature_count, action_dim, log_std=-3, max_log_std=0.45, device='cpu'):
        super(Actor, self).__init__()

        self.device = device

        self.action_dim = action_dim
        self.max_log_std = max_log_std
        self.feature_count = feature_count

        self.hidden_dims = 2048
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        self.num_layers = 1
        self.max_action = 1.0

        # Linear layer to decode the static features of the state
        self.fc0 = nn.Linear(feature_count, self.hidden_dims)
        self.fc1 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims_halved)
        # self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims)
        # self.fc4 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc5 = nn.Linear(self.hidden_dims_halved, self.hidden_dims_halved)
        self.fc6 = nn.Linear(self.hidden_dims_halved, self.hidden_dims_halved // 2)

        self.decoder = nn.Linear(self.hidden_dims_halved // 2, self.action_dim)

        self.log_std = nn.Linear(self.hidden_dims_halved // 2, self.action_dim)

        # self.init_weights()

        self.to(device=self.device)
        self.to(dtype=torch.bfloat16)

        # Initialize weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Cast to Float32 before applying orthogonal initialization
                weight = m.weight.data.to(torch.float32)
                nn.init.orthogonal_(weight)
                m.weight.data = weight.to(m.weight.dtype)  # Convert back to original dtype
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.fc0.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        # self.fc3.apply(init_weights)
        # self.fc4.apply(init_weights)
        self.fc5.apply(init_weights)
        self.fc6.apply(init_weights)

        # self.raycast.apply(init_weights)
        self.decoder.apply(init_weights)

    def reset_lstm(self):
        self.hidden_state = torch.zeros(self.num_layers, 1, self.hidden_dims, dtype=torch.bfloat16, device=self.device)
        self.cell_state = torch.zeros(self.num_layers, 1, self.hidden_dims, dtype=torch.bfloat16, device=self.device)

    def forward(self, state, mask=None, hidden_state=None, cell_state=None):
        """
        Input shape: (batch_size, sequence_length, feature_count)
        """
        # Encode the static features of the state
        # x = F.leaky_relu(self.fc0(state[:, :, :self.feature_count - (128 + 64*3)]), 0.01)

        # The raycasting parts of the state gets pre-processed by a separate sequence of linear layers
        # raycasting_data = state[:, :, -(128 + 64*3):]  # Reshape to (batch_size*sequence_length, channels, height, width)

        # Process all raycasting data through the raycasting network
        # raycast_out = self.raycast(raycasting_data)

        # Concatenate the raycast output with the non-raycasting part of state
        # x = torch.cat((x, raycast_out), dim=2)

        # x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))

        # x = x[:, -1, :]
        x = state[:, -1, :]

        x = F.leaky_relu(self.fc0(x), 0.01)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        # x = F.leaky_relu(self.fc3(x), 0.01)
        # x = F.leaky_relu(self.fc4(x), 0.01)
        x = F.leaky_relu(self.fc5(x), 0.01)
        x = F.leaky_relu(self.fc6(x), 0.01)

        mu = F.tanh(self.decoder(x))

        log_std = F.softplus(self.log_std(x))
        log_std = torch.clamp(log_std, -20, self.max_log_std)

        # if mask is not None:
        #     mu = mu * mask
        #     log_std = log_std * mask + 1e-5

        return mu, log_std, hidden_state, cell_state

    def get_action(self, state, action=None, mask=None, hidden_state=None, cell_state=None):
        # If mask is not tensor, spit warning once and convert it to tensor
        if mask is not None and not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.bfloat16, device=self.device)

            if self._warned_mask_is_not_tensor is None:
                self._warned_mask_is_not_tensor = True
                print("Warning: Mask is not a tensor, converting it to tensor")

        action_mean, log_std, hidden_state, cell_state = self(state, mask=None, hidden_state=hidden_state, cell_state=cell_state)
        action_std = log_std

        probs = Normal(action_mean, action_std)

        if action is None:
            # self.hidden_state = hidden_state
            # self.cell_state = cell_state
            action = probs.sample()

        return action, probs, action_mean, log_std, (hidden_state, cell_state)


class Critic(nn.Module):
    def __init__(self, feature_count, action_dim, device='cpu'):
        super(Critic, self).__init__()
        self.device = device

        self.action_dim = action_dim
        self.feature_count = feature_count

        self.hidden_dims = 2048
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        # Linear layer to decode the static features of the state
        self.fc0 = nn.Linear(feature_count, self.hidden_dims)
        self.fc1 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims_halved)
        # self.fc3 = nn.Linear(self.hidden_dims, self.hidden_dims)
        # self.fc4 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc5 = nn.Linear(self.hidden_dims_halved, self.hidden_dims_halved)
        self.fc6 = nn.Linear(self.hidden_dims_halved, self.hidden_dims_halved // 2)

        self.decoder = nn.Linear(self.hidden_dims_halved // 2, self.action_dim)

        self.to(device=self.device)
        self.to(dtype=torch.bfloat16)

        # Initialize weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Cast to Float32 before applying orthogonal initialization
                weight = m.weight.data.to(torch.float32)
                nn.init.orthogonal_(weight)
                m.weight.data = weight.to(m.weight.dtype)  # Convert back to original dtype
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.fc0.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        # self.fc3.apply(init_weights)
        # self.fc4.apply(init_weights)
        self.fc5.apply(init_weights)
        self.fc6.apply(init_weights)

        # self.raycast.apply(init_weights)
        # self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        # # Layer normalization for transformer encoder layers
        # for layer in self.transformer_encoder.layers:
        #     nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
        #     nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
        #     nn.init.xavier_uniform_(layer.linear1.weight)
        #     nn.init.xavier_uniform_(layer.linear2.weight)
        #     nn.init.constant_(layer.self_attn.in_proj_bias, 0)
        #     nn.init.constant_(layer.self_attn.out_proj.bias, 0)
        #     nn.init.constant_(layer.linear1.bias, 0)
        #     nn.init.constant_(layer.linear2.bias, 0)

    def forward(self, state, hidden_states=None, cell_states=None):
        """
        Input shape: (batch_size, sequence_length, feature_count)
        """
        # state = state[:, -1, :]

        # Encode the static features of the state
        # x = F.leaky_relu(self.fc0(state[:, :, :self.feature_count - (128 + 64*3)]), 0.01)
        #
        # # The raycasting parts of the state gets pre-processed by a separate sequence of linear layers
        # # The raycasting parts of the state gets pre-processed by a separate sequence of linear layers
        # raycasting_data = state[:, :, -(128 + 64*3):].view(-1, 5, 8, 8)  # Reshape to (batch_size*sequence_length, channels, height, width)
        #
        # # Process all raycasting data through the raycasting network
        # raycast_out = self.raycast(raycasting_data)
        # raycast_out = raycast_out.view(state.size(0), state.size(1),
        #                                -1)  # Reshape back to (batch_size, sequence_length, features)
        #
        # # Concatenate the raycast output with the non-raycasting part of state
        # x = torch.cat((x, raycast_out), dim=2)

        # x, _ = self.lstm(x, (hidden_states, cell_states))

        # x = x[:, -1, :]
        x = state[:, -1, :]

        x = F.leaky_relu(self.fc0(x), 0.01)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        # x = F.leaky_relu(self.fc3(x), 0.01)
        # x = F.leaky_relu(self.fc4(x), 0.01)
        x = F.leaky_relu(self.fc5(x), 0.01)
        x = F.leaky_relu(self.fc6(x), 0.01)

        # Transformer
        # x = self.encoder(x)
        # x = self.transformer_encoder(x)

        return self.decoder(x)
