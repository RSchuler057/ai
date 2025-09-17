import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List

# simple actor-critic network (policy logits + value)
class PokerNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)

class PPOTrainer:
    def __init__(self, obs_dim, n_actions, lr=3e-4, clip=0.2, gamma=0.99, lmbda=0.95, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PokerNet(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.clip = clip
        self.gamma = gamma
        self.lmbda = lmbda
        self.n_actions = n_actions

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(obs_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        act = dist.sample()
        return int(act.item()), float(dist.log_prob(act).item()), float(value.item())
    
    def update(self, trajectories: List[dict], epochs=4, batch_size=64):
        """
        trajectories: list of dicts with keys 'obs', 'action', 'logp', 'value', 'reward', 'done'
        Compute GAE advantages and optimize PPO objective
        """

        # flatten
        obs = np.array([t['obs'] for t in trajectories], dtype=np.float32)
        actions = np.array([t['action'] for t in trajectories], dtype=np.int64)
        old_logps = np.array([t['logp'] for t in trajectories], dtype=np.float32)
        values = np.array([t['value'] for t in trajectories], dtype=np.float32)
        rewards = np.array([t['reward'] for t in trajectories], dtype=np.float32)
        dones = np.array([t['done'] for t in trajectories], dtype=np.float32)

        # compute return and advantages with GAE
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        lastgaelam = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            lastgaelam = delta + self.gamma * self.lmbda * mask * lastgaelam
            advantages[t] = lastgaelam
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8) # normalize

        dataset_size = len(obs)
        idxs = np.arange(dataset_size)
        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, batch_size):
                mb_idxs = idxs[start:start+batch_size]
                logits, values_pred = self.net(obs_t[mb_idxs])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logps = dist.log_prob(actions_t[mb_idxs])
                ratio = torch.exp(new_logps - old_logps_t[mb_idxs])
                surr1 = ratio * adv_t[mb_idxs]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb_idxs]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns_t[mb_idxs] - values_pred).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()