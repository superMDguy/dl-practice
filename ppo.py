"""
PPO Implementation

References:
    - https://arxiv.org/pdf/1707.06347
    - https://arxiv.org/pdf/1506.02438
    - https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    - https://arxiv.org/pdf/2006.05990
    - https://github.com/vwxyzjn/cleanrl
"""

import warnings

warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated as an API", category=UserWarning
)

import argparse
import multiprocessing as mp
import random
import time
from contextlib import contextmanager
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

SEED = 1337
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


ENV_NAME = "LunarLander-v3"
ENV_STATE_DIM = 8
ENV_ACTION_DIM = 4


@contextmanager
def get_env(render_mode=None):
    env = gym.make(ENV_NAME, render_mode=render_mode)
    try:
        yield env
    finally:
        env.close()


# https://github.com/vwxyzjn/cleanrl/blob/004f8a086a892a2a180f4dd332b90d83a968aa7a/cleanrl/ppo.py#L94
@torch.no_grad()
def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.policy_logits = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        self.value = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def __call__(self, x: Tensor) -> Tuple[Categorical, Tensor]:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, Tensor]:
        h = self.trunk(x)
        logits = self.policy_logits(h)
        dist = Categorical(logits=logits)
        value = self.value(h).squeeze(-1)  # (B,)
        return dist, value


def ppo_loss(
    model: ActorCritic,
    eps: float,
    action_logp: Tensor,  # (B, )
    state: Tensor,  # (B, state)
    action: Tensor,  # (B, )
    advantage: Tensor,  # (B, )
    value_target: Tensor,  # (B, )
    c1: float = 0.5,
    c2: float = 0.01,
) -> Tuple[Tensor, Tensor]:  # (B, ); (B, )
    # Clip Loss
    actions_dist, value = model(state)
    actions_logp = actions_dist.log_prob(action)  # (B, )
    ratio = torch.exp(actions_logp - action_logp)  # (B, )

    clip_loss = -torch.min(
        ratio * advantage,
        torch.clamp(ratio, 1 - eps, 1 + eps) * advantage,
    ).mean()
    value_loss = F.mse_loss(value, value_target)
    entropy_loss = -actions_dist.entropy().mean()

    return clip_loss + c1 * value_loss + c2 * entropy_loss, ratio


def normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    return (x - x.mean()) / (x.std() + eps)


def rollout(
    model: ActorCritic,
    T: int,
    gamma: float,
    lambd: float,
) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, float
]:  # (states, actions, action_logps, values, advantages)
    states = np.zeros((T, ENV_STATE_DIM), dtype=np.float32)
    actions = np.zeros(T, dtype=np.int64)
    action_logps = np.zeros(T, dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    value_ests = np.zeros(T, dtype=np.float32)
    dones = np.zeros(T, dtype=bool)

    with get_env() as env:
        observation, _ = env.reset()
        for t in range(T):
            with torch.no_grad():
                action_dist, pred_value = model(torch.from_numpy(observation))
                action = action_dist.sample()

            action_logps[t] = action_dist.log_prob(action)
            action = action.item()

            states[t] = observation
            actions[t] = action
            value_ests[t] = pred_value.item()

            observation, reward, terminated, truncated, _ = env.step(action)

            rewards[t] = reward
            if truncated or terminated:
                dones[t] = True
                observation, _ = env.reset()
    avg_episode_reward = rewards.sum() / max(dones.sum(), 1)

    with torch.no_grad():
        _, last_pred_value = model(torch.tensor(observation))
        last_pred_value = last_pred_value.item()

    advantages = np.zeros(T, dtype=np.float32)
    # Special case for final state (no future rewards)
    if not dones[T - 1]:
        advantages[T - 1] = rewards[T - 1] + gamma * last_pred_value - value_ests[T - 1]
    else:
        advantages[T - 1] = rewards[T - 1] - value_ests[T - 1]

    for t in reversed(range(T - 1)):
        if not dones[t]:
            # TD error: how much does value est. differ from Bellman equation
            delta_t = rewards[t] + gamma * value_ests[t + 1] - value_ests[t]
            # recursive formulation of GAE
            advantages[t] = delta_t + gamma * lambd * advantages[t + 1]
        else:  # episode over, future rewards aren't relevant
            advantages[t] = rewards[t] - value_ests[t]

    values = advantages + value_ests

    return (
        torch.from_numpy(states),
        torch.from_numpy(actions),
        torch.from_numpy(action_logps),
        torch.from_numpy(values),
        torch.from_numpy(advantages),
        avg_episode_reward,
    )


def train(
    n_rollouts: int = 64,
    n_actors=16,
    T=512,
    gamma=0.99,
    lambd=0.9,
    bs=256,
    eps: float = 0.25,
):
    model = ActorCritic(ENV_STATE_DIM, ENV_ACTION_DIM)
    model.compile()
    # TODO: derive proper LR?
    optim = torch.optim.Adam(model.parameters(), lr=8e-4)
    # Derived from clip param: avg per-sample KL when the ratio sits at the clip boundary [1-eps, 1+eps].
    kl_target = -np.log(1 - eps**2) / 2

    for i in range(n_rollouts):
        print(f"\n=== ROLLOUT {i + 1} ===")
        rollout_start = time.perf_counter()
        with mp.Pool(n_actors) as pool:
            results = pool.starmap(rollout, [(model.cpu(), T, gamma, lambd)] * n_actors)

        (
            all_states,
            all_actions,
            all_action_logps,
            all_values,
            all_advantages,
            all_avg_rewards,
        ) = zip(*results)

        all_states = torch.cat(all_states).to(device)
        all_actions = torch.cat(all_actions).to(device)
        all_action_logps = torch.cat(all_action_logps).to(device)
        all_values = torch.cat(all_values).to(device)
        all_advantages = normalize(torch.cat(all_advantages)).to(device)

        print(f"Avg. reward: {np.mean(all_avg_rewards):.2f}")

        rollout_secs = time.perf_counter() - rollout_start
        n_steps = n_actors * T
        print(
            f"[perf] simulated {n_steps} steps in {rollout_secs:.2f} secs (avg. {rollout_secs / n_steps * 1000:.3f} ms/step) "
        )

        model = model.to(device)
        model.train()
        # Create a simple dataset and dataloader for batching
        dataset = torch.utils.data.TensorDataset(
            all_states, all_actions, all_action_logps, all_advantages, all_values
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

        j = 0
        approx_kl = 0.0
        train_start = time.perf_counter()
        while approx_kl < kl_target:
            running_loss = 0.0
            kl_estimates = []
            for (
                batch_states,
                batch_actions,
                batch_action_logps,
                batch_advantages,
                batch_values,
            ) in dataloader:
                optim.zero_grad()
                loss, ratio = ppo_loss(
                    model,
                    eps,
                    batch_action_logps,
                    batch_states,
                    batch_actions,
                    batch_advantages,
                    batch_values,
                )
                kl_estimates.append(
                    # http://joschu.net/blog/kl-approx.html
                    ((ratio - 1) - torch.log(ratio)).mean().item()
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optim.step()
                running_loss += loss.item()

            approx_kl = np.mean(kl_estimates)
            print(
                f"\tEpoch {j + 1}, loss {(running_loss / len(dataloader)):.2f}, kl {approx_kl:.3f}"
            )
            j += 1
        train_secs = time.perf_counter() - train_start
        print(
            f"[perf] trained {j} epochs in {train_secs:.2f} seconds (avg. {train_secs / j:.3f} secs/epoch)"
        )

    print("\n=== TRAINING COMPLETE — running trained policy ===")
    run_trained(model)


def run_trained(model: ActorCritic, n_episodes: int = 5):
    model = model.cpu()
    model.eval()
    with get_env(render_mode="human") as env:
        for episode in range(n_episodes):
            observation, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    action_dist, _ = model(torch.from_numpy(observation))
                    action = action_dist.sample().item()
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            print(f"Episode {episode + 1}: total reward = {total_reward:.1f}")


def run_random():
    with get_env(render_mode="human") as env:
        observation, info = env.reset(seed=SEED)
        print("init obs", observation, type(observation))
        print("init info", info)
        for _ in range(1000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="Run random policy")
    parser.add_argument("--train", action="store_true", help="Train")
    args = parser.parse_args()

    if args.random:
        run_random()
    elif args.train:
        train()
    else:
        parser.print_help()
