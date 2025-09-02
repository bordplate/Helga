import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


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
        action, probs, mu, log_std = self.actor.get_action(state, None, mask=mask)

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

        self.hidden_dims = 2048
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        self.num_layers = 1
        self.max_action = 1.0

        # Linear layer to decode the static features of the state
        self.fc0 = nn.Linear(feature_count, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 1536)

        self.decoder = nn.Linear(1536, self.action_dim)
        self.log_std = nn.Linear(1536, self.action_dim)

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

        self.decoder.apply(init_weights)

    def forward(self, state, mask=None):
        """
        Input shape: (batch_size, sequence_length, feature_count)
        """
        x = state

        x = F.leaky_relu(self.fc0(x), 0.01)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)

        mu = F.tanh(self.decoder(x))

        log_std = F.softplus(self.log_std(x))
        log_std = torch.clamp(log_std, -20, self.max_log_std)

        # if mask is not None:
        #     mu = mu * mask
        #     log_std = log_std * mask + 1e-5

        return mu, log_std

    def get_action(self, state, action=None, mask=None):
        # If mask is not tensor, spit warning once and convert it to tensor
        if mask is not None and not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.bfloat16, device=self.device)

            if self._warned_mask_is_not_tensor is None:
                self._warned_mask_is_not_tensor = True
                print("Warning: Mask is not a tensor, converting it to tensor")

        action_mean, log_std = self(state, mask=None)
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

        self.hidden_dims = 2048
        self.hidden_dims_halved = int(self.hidden_dims / 2)

        # Linear layer to decode the static features of the state
        self.fc0 = nn.Linear(feature_count, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 1536)

        self.decoder = nn.Linear(1536, self.action_dim)

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

        self.decoder.apply(init_weights)

    def forward(self, state):
        """
        Input shape: (batch_size, sequence_length, feature_count)
        """
        x = state

        x = F.leaky_relu(self.fc0(x), 0.01)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)

        # Transformer
        # x = self.encoder(x)
        # x = self.transformer_encoder(x)

        return self.decoder(x)
