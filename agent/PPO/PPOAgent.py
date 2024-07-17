import numba
import torch
import torch.nn as nn
import numpy as np

from PPO.ActorCritic import ActorCritic
from RolloutBuffer import RolloutBuffer
from RE3.RandomEncoder import RandomEncoder

class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 buffer_size=1024 * 32,
                 batch_size=1024 * 8,
                 mini_batch_size=1024,
                 gamma=0.99,
                 K_epochs=5,
                 eps_clip=0.2,
                 log_std=-3.0,
                 ent_coef=0.01,
                 cl_coeff=0.5,
                 max_grad_norm=0.5,
                 beta=0.1,
                 kl_threshold=0.1,
                 lambda_gae=0.95,
                 device='cpu',
                 buffer: RolloutBuffer | None = None,
                 ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef
        self.cl_coeff = cl_coeff
        self.max_grad_norm = max_grad_norm
        self.beta = beta
        self.kl_threshold = kl_threshold
        self.lambda_gae = lambda_gae

        self.device = device

        self.buffer = buffer
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.batch_size = batch_size
        self.replay_buffers = []

        self.random_encoder = RandomEncoder(state_dim).to(device)

        self.policy = ActorCritic(state_dim, action_dim, log_std, device)

        self.action_mask = torch.ones(action_dim).to(device)

        self.optimizer = torch.optim.AdamW([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ], eps=1e-5, weight_decay=1e-4)

        self.mse_loss = nn.MSELoss()

    def start_new_episode(self):
        self.policy.start_new_episode()

    def load_policy_dict(self, policy):
        self.policy.load_state_dict(policy)

    def choose_action(self, state_sequence: torch.Tensor):
        with torch.no_grad():
            self.policy.eval()

            action, action_logprob, state_value = self.policy.act(state_sequence, self.action_mask)

        return action, action_logprob, state_value

    @torch.no_grad()
    def compute_returns_and_advantages(self,
                                       rewards: torch.Tensor,
                                       dones: torch.Tensor,
                                       state_values: torch.Tensor,
                                       last_value: torch.bfloat16,
                                       done: torch.bool) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert tensors to numpy arrays
        rewards_np = rewards.to(dtype=torch.float32).cpu().numpy()
        dones_np = dones.to(dtype=torch.float32).cpu().numpy()
        state_values_np = state_values.to(dtype=torch.float32).cpu().numpy()
        last_value_np = last_value.to(dtype=torch.float32).cpu().numpy()

        # Call the Numba-optimized function
        advantages_np = numba_compute_advantages(rewards_np, dones_np, state_values_np,
                                                 self.gamma, self.lambda_gae, last_value_np[0], float(done.cpu().numpy()))

        # Convert back to tensors
        advantages = torch.tensor(advantages_np, dtype=torch.bfloat16, device=self.device)
        returns = advantages + state_values

        return advantages, returns

    # @torch.no_grad()
    # def compute_returns_and_advantages(self,
    #                                    rewards: torch.Tensor,
    #                                    dones: torch.Tensor,
    #                                    state_values: torch.Tensor,
    #                                    last_value: torch.bfloat16,
    #                                    done: torch.bool) -> tuple[torch.Tensor, torch.Tensor]:
    #     # Initialize tensors
    #     advantages = torch.zeros_like(rewards).to(self.device)
    #
    #     # Precompute masks
    #     masks = 1.0 - dones.float()
    #     masks = torch.cat((masks[1:], torch.tensor([1.0 - float(done)], device=self.device)))
    #
    #     # Initialize the last_gae_lam and next_value
    #     last_gae_lam = torch.zeros(1, device=self.device)
    #     next_value = last_value * (1.0 - float(done))
    #
    #     # Compute the deltas and masks
    #     deltas = rewards + self.gamma * torch.cat(
    #         (state_values[1:] * masks[:-1], torch.tensor([next_value], device=self.device))) - state_values
    #
    #     # Reverse accumulate the advantages
    #     for step in reversed(range(len(rewards))):
    #         last_gae_lam = deltas[step] + self.gamma * self.lambda_gae * last_gae_lam * masks[step]
    #         advantages[step] = last_gae_lam
    #
    #     # Compute returns
    #     returns = advantages + state_values
    #
    #     return advantages, returns

    def learn(self):
        self.policy.train()
        self.policy.actor.train()
        self.policy.critic.train()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_approx_kl = 0

        count = 0

        old_params = self.policy.actor.named_parameters()
        old_params = {k: v.clone() for k, v in old_params}

        # Optimize policy for K epochs
        for epoch in range(self.K_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                self.optimizer.zero_grad()

                for slice in range(0, self.batch_size, self.mini_batch_size):
                    batch_slice = [item[slice:slice + self.mini_batch_size] for item in batch]

                    (states, actions, _rewards, dones, old_logprobs) = zip(batch_slice)
                     # cell_states, hidden_states, advantages, returns) = zip(batch_slice)
                     # advantages, returns) = zip(batch_slice)

                    states = states[0].clone().to(self.device)
                    actions = actions[0].clone().to(self.device)
                    old_logprobs = old_logprobs[0].clone().to(self.device)
                    _rewards = _rewards[0].clone().to(self.device)
                    dones = dones[0].clone().to(self.device)

                    # advantages = advantages[0].clone().to(self.device)
                    # returns = returns[0].clone().to(self.device)
                    # hidden_states = hidden_states[0].clone().permute(1, 0, 2).to(self.device)
                    # cell_states = cell_states[0].clone().permute(1, 0, 2).to(self.device)

                    # Evaluating old actions and values
                    logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions, self.action_mask, None, None)

                    advantages, returns = self.compute_returns_and_advantages(
                        _rewards.detach(), dones.detach(), state_values.squeeze().detach(), state_values[-1].detach(), dones[-1].detach()
                    )

                    # source_y_t = self.random_encoder(states[:, -1, :].squeeze())

                    # intrinsic_rewards = self.random_encoder.compute_intrinsic_rewards(source_y_t[:, :, 0, 0], y_t.squeeze(), True)
                    # intrinsic_rewards = intrinsic_rewards.squeeze().to(device)

                    old_logprobs = old_logprobs.squeeze(dim=1).detach()

                    # advantages = advantages + self.beta * intrinsic_rewards
                    # returns = returns + self.beta * intrinsic_rewards

                    # Finding the ratio (pi_theta / pi_theta__old)
                    ratios = torch.exp(logprobs.sum(dim=1) - old_logprobs.sum(dim=1))
                    # ratios = torch.exp(logprobs - old_logprobs.squeeze(dim=1).detach())

                    # Normalizing the advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

                    surr1 = advantages * ratios
                    surr2 = advantages * torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)

                    actor_loss = -(torch.min(surr1, surr2)).mean()
                    critic_loss = self.mse_loss(state_values.squeeze(), returns.squeeze())

                    total_policy_loss += actor_loss.item()
                    total_value_loss += critic_loss.item()

                    entropy_loss = dist_entropy.sum(dim=1).mean()
                    total_entropy_loss += entropy_loss.item()

                    approx_kl = -((logprobs - old_logprobs).mean())
                    total_approx_kl += abs(approx_kl.item())
                    count += 1

                    loss = actor_loss - self.ent_coef * entropy_loss + self.cl_coeff * critic_loss

                    loss.backward()
                    total_loss += loss.item()

                    total_loss += loss.item()

                    # Stop early if kl_divergence is above threshold
                    if total_approx_kl / count > self.kl_threshold:
                        print(f"Stopping early in {epoch}:{count} due to high KL divergence: {total_approx_kl / count }")
                        break

                # Clip the gradients
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)

                self.optimizer.step()

            # Stop early if kl_divergence is above threshold
            if total_approx_kl / count > self.kl_threshold:
                break

        torch.cuda.empty_cache()

        # Print difference for all parameters
        for name, param in self.policy.actor.named_parameters():
            if name in old_params:
                diff = (param - old_params[name]).abs().sum().item()
                print(f"Param: {name}, diff: {diff}")

        self.buffer.clear()

        return (total_loss / count if count > 0 else 0,
                total_policy_loss / count if count > 0 else 0,
                total_value_loss / count if count > 0 else 0,
                total_entropy_loss / count if count > 0 else 0,
                total_approx_kl / count if count > 0 else 0)


@numba.jit(nopython=True)
def numba_compute_advantages(rewards, dones, state_values, gamma, lambda_gae, last_value, done):
    advantages = np.zeros_like(rewards)
    masks = 1.0 - dones
    masks = np.append(masks[1:], 1.0 - done)
    next_value = last_value * (1.0 - done)
    deltas = rewards + gamma * np.append(state_values[1:] * masks[:-1], next_value) - state_values

    last_gae_lam = 0
    for step in range(len(rewards) - 1, -1, -1):  # Manual reverse
        last_gae_lam = deltas[step] + gamma * lambda_gae * last_gae_lam * masks[step]
        advantages[step] = last_gae_lam

    return advantages