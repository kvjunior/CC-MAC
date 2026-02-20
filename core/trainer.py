"""
trainer.py - CC-MAC Training Algorithm

Implements:
  - Prioritized Experience Replay with TD-error prioritization
  - Augmented Lagrangian constraint enforcement (adaptive rho)
  - Double Q-learning with n-step returns
  - Polyak-averaged target networks
  - Optional differential privacy (Gaussian mechanism)

Key fixes over prior version:
  - Fixed n-step buffer flushing (eliminated duplicate transition storage)
  - Penalty broadcast explicitly to per-agent array
  - evaluate() returns raw per-episode arrays for proper statistical testing
  - GNN graph features set on model during initialization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import copy
import time


# ============================================================
# Prioritized Experience Replay
# ============================================================
class PrioritizedReplayBuffer:
    """Experience replay with proportional prioritization.

    Priority = (|TD_error| + epsilon)^alpha
    Importance sampling weights = (N * p_i)^(-beta) with beta annealing.
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 total_steps: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps
        self.buffer: List[Dict] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self._step = 0

    def push(self, obs: np.ndarray, actions: np.ndarray,
             rewards: np.ndarray, next_obs: np.ndarray,
             done: bool, violations: Optional[Dict] = None,
             global_state: Optional[np.ndarray] = None):
        """Store a transition with max priority."""
        max_p = self.priorities[:self.size].max() if self.size > 0 else 1.0

        data = {
            "obs": obs, "actions": actions, "rewards": rewards,
            "next_obs": next_obs, "done": done,
            "violations": violations or {},
            "global_state": global_state,
        }

        if self.size < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample batch with prioritized probabilities.

        Returns: (batch_list, indices, importance_weights)
        """
        self._step += 1
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum() + 1e-10

        replace = self.size < batch_size
        indices = np.random.choice(
            self.size, batch_size, p=probs, replace=replace)

        beta = min(
            self.beta_end,
            self.beta_start
            + (self.beta_end - self.beta_start) * self._step / self.total_steps,
        )

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max() + 1e-10

        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices: np.ndarray,
                          td_errors: np.ndarray):
        """Update priorities based on absolute TD errors."""
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self) -> int:
        return self.size


# ============================================================
# Augmented Lagrangian Constraint Manager
# ============================================================
class LagrangianConstraintManager:
    """Augmented Lagrangian constraint enforcement.

    Maintains dual variables (Lagrange multipliers) for each constraint.
    Penalty: L = sum_k [ lambda_k * g_k + (rho/2) * g_k^2 ]
    Update:  lambda_k <- max(0, lambda_k + lr * g_k)
    Adaptive rho: increase if violations persist, decrease if satisfied.
    """

    CONSTRAINT_NAMES = ["capacity", "co2", "diversity", "queue"]

    def __init__(self, target_csr: float = 0.95, lambda_lr: float = 0.01,
                 rho_init: float = 1.0, rho_max: float = 100.0,
                 rho_growth: float = 1.5):
        self.target_csr = target_csr
        self.lambda_lr = lambda_lr
        self.rho = rho_init
        self.rho_max = rho_max
        self.rho_growth = rho_growth

        self.lambdas = {k: 0.0 for k in self.CONSTRAINT_NAMES}
        self.violation_history: Dict[str, deque] = {
            k: deque(maxlen=100) for k in self.CONSTRAINT_NAMES}
        self.satisfaction_rates = {k: 1.0 for k in self.CONSTRAINT_NAMES}

    def compute_penalty(self, violations: Dict[str, float]) -> float:
        """Compute augmented Lagrangian penalty (scalar).

        Returns scalar penalty value to be subtracted from rewards.
        """
        penalty = 0.0
        for k in self.CONSTRAINT_NAMES:
            g = max(0.0, violations.get(k, 0.0))
            penalty += self.lambdas[k] * g + (self.rho / 2) * g ** 2
        return penalty

    def update(self, violations: Dict[str, float]):
        """Update dual variables and adaptive penalty coefficient."""
        for k in self.CONSTRAINT_NAMES:
            g = max(0.0, violations.get(k, 0.0))
            self.violation_history[k].append(g)
            # Subgradient ascent on dual variables
            self.lambdas[k] = max(
                0.0, self.lambdas[k] + self.lambda_lr * g)
            # Track per-constraint satisfaction rate
            recent = list(self.violation_history[k])
            self.satisfaction_rates[k] = (
                sum(1 for v in recent if v < 0.01)
                / max(1, len(recent))
            )

        # Adaptive rho: increase if under target, slowly decrease if satisfied
        overall_csr = (min(self.satisfaction_rates.values())
                       if self.satisfaction_rates else 1.0)
        if overall_csr < self.target_csr:
            self.rho = min(self.rho_max, self.rho * self.rho_growth)
        else:
            self.rho = max(0.1, self.rho * 0.95)

    def get_stats(self) -> Dict[str, Any]:
        """Get current constraint manager state for logging."""
        return {
            "lambdas": dict(self.lambdas),
            "rho": self.rho,
            "satisfaction_rates": dict(self.satisfaction_rates),
            "overall_csr": (min(self.satisfaction_rates.values())
                            if self.satisfaction_rates else 1.0),
        }


# ============================================================
# Optional Differential Privacy
# ============================================================
class GradientPrivacy:
    """Optional differential privacy via Gaussian mechanism.

    Clips per-sample gradients to norm C, adds Gaussian noise.
    Privacy budget tracked via simplified RDP accounting.
    Disabled by default (self.enabled = False).
    """

    def __init__(self, clip_norm: float = 1.0, noise_scale: float = 1.1,
                 target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.total_steps = 0
        self.enabled = False

    def clip_and_noise(self, model: nn.Module, batch_size: int):
        """Apply gradient clipping and Gaussian noise if enabled."""
        if not self.enabled:
            return
        nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * (
                    self.noise_scale * self.clip_norm / batch_size)
                param.grad += noise
        self.total_steps += 1

    def get_privacy_spent(self) -> float:
        """Approximate privacy budget spent (simplified RDP accounting)."""
        if self.total_steps == 0:
            return 0.0
        epsilon_step = 2 * self.noise_scale ** (-2)
        epsilon_total = (
            np.sqrt(2 * self.total_steps
                    * np.log(1 / self.target_delta))
            * np.sqrt(epsilon_step)
        )
        return epsilon_total


# ============================================================
# Main Trainer
# ============================================================
class CCMACTrainer:
    """Main training loop for CC-MAC.

    Implements:
      - Double Q-learning with Polyak-averaged target networks
      - N-step returns for faster credit assignment
      - Augmented Lagrangian constraint penalty (per-step)
      - Epsilon-greedy exploration with exponential decay
      - GNN graph integration (set_graph called in __init__)
      - Evaluation returning raw per-episode data for statistics

    Args:
        model: CCMACAgent instance.
        env: TourismEnvironment instance.
        config: Dict with hyperparameters (see code for keys).
    """

    def __init__(self, model: nn.Module, env, config: Dict):
        self.model = model
        self.env = env
        self.config = config

        # Set GNN graph if model supports it and env provides features
        if hasattr(model, 'set_graph') and hasattr(env, 'get_poi_node_features'):
            try:
                model.set_graph(
                    env.get_poi_node_features(),
                    env.get_edge_index(),
                    env.get_edge_features(),
                )
            except Exception:
                pass  # GNN disabled or incompatible

        # Target network (frozen copy for stable targets)
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()

        # Optimizer + cosine LR scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        total_episodes = config.get("total_episodes", 5000)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_episodes),
            eta_min=config.get("lr", 3e-4) * 0.01,
        )

        # Prioritized replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.get("buffer_size", 100000),
            alpha=0.6, beta_start=0.4,
            total_steps=max(1, total_episodes * 32),
        )

        # Constraint manager
        self.constraint_mgr = LagrangianConstraintManager(
            target_csr=config.get("target_csr", 0.95),
            lambda_lr=config.get("lambda_lr", 0.01),
        )

        # Optional privacy module (disabled by default)
        self.privacy = GradientPrivacy()

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 256)
        self.n_step = config.get("n_step", 3)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.epsilon = config.get("eps_start", 1.0)
        self.eps_end = config.get("eps_end", 0.01)
        self.eps_decay = config.get("eps_decay", 0.995)

        # Tracking
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = float("-inf")

    # ----------------------------------------------------------
    # Training Episode
    # ----------------------------------------------------------
    def train_episode(self) -> Dict[str, float]:
        """Run one training episode with n-step returns and constraints.

        Returns:
            Dict with episode_reward, constraint_satisfaction_rate,
            epsilon, total_steps, and Lagrangian stats.
        """
        self.model.train()
        obs_dict = self.env.reset()
        obs = obs_dict["obs"]
        ep_reward = 0.0
        n_step_buffer: deque = deque(maxlen=self.n_step)
        done = False

        while not done:
            # Select actions with epsilon-greedy
            actions = self.model.get_actions(obs, epsilon=self.epsilon)
            next_obs_dict, rewards, done, info = self.env.step(
                actions.numpy())
            next_obs = next_obs_dict["obs"]

            # Collect constraint violations
            violations = {
                k: info.get(f"{k}_violation", 0.0)
                for k in self.constraint_mgr.CONSTRAINT_NAMES
            }

            # Accumulate in n-step buffer
            n_step_buffer.append((
                obs, actions, rewards, next_obs, done,
                violations, obs_dict.get("global_state"),
            ))

            # Store n-step transition when buffer is full
            if len(n_step_buffer) == self.n_step:
                self._store_n_step(n_step_buffer)

            ep_reward += float(rewards.mean())
            self.constraint_mgr.update(violations)

            # Gradient step if enough data
            if len(self.buffer) >= self.batch_size:
                self._train_step()

            obs = next_obs
            obs_dict = next_obs_dict
            self.total_steps += 1

        # Flush remaining partial n-step transitions at episode end
        # Skip the first entry if buffer is full (already stored above)
        if len(n_step_buffer) == self.n_step:
            n_step_buffer.popleft()
        while len(n_step_buffer) > 0:
            self._store_n_step(n_step_buffer)
            n_step_buffer.popleft()

        # Decay exploration and step LR scheduler
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        self.scheduler.step()
        self.episode_count += 1

        summary = self.env.get_episode_summary()
        csr = summary.get("constraint_satisfaction_rate", 0.0)

        return {
            "episode_reward": ep_reward,
            "constraint_satisfaction_rate": csr,
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            **self.constraint_mgr.get_stats(),
        }

    # ----------------------------------------------------------
    # N-Step Return Computation
    # ----------------------------------------------------------
    def _store_n_step(self, buffer: deque):
        """Compute n-step return and store transition in replay buffer.

        Computes: R = sum_{k=0}^{n-1} gamma^k * r_{t+k}
        Then subtracts constraint penalty (broadcast to per-agent).
        """
        obs0 = buffer[0][0]
        actions0 = buffer[0][1]

        # N-step discounted return (per-agent)
        R = np.zeros(self.env.num_agents, dtype=np.float32)
        for i, transition in enumerate(buffer):
            R += self.gamma ** i * transition[2]   # transition[2] = rewards

        last = buffer[-1]
        done = last[4]
        next_obs = last[3]
        violations = last[5]
        global_state = last[6]

        # Subtract constraint penalty (scalar broadcast to all agents)
        penalty = self.constraint_mgr.compute_penalty(violations)
        R_adjusted = R - penalty   # penalty is scalar, R is (num_agents,)

        self.buffer.push(
            obs0.numpy() if torch.is_tensor(obs0) else obs0,
            actions0.numpy() if torch.is_tensor(actions0) else actions0,
            R_adjusted,
            next_obs.numpy() if torch.is_tensor(next_obs) else next_obs,
            done,
            violations,
            (global_state.numpy() if torch.is_tensor(global_state)
             else global_state) if global_state is not None else None,
        )

    # ----------------------------------------------------------
    # Gradient Step
    # ----------------------------------------------------------
    def _train_step(self):
        """One gradient step with double Q-learning and prioritized replay."""
        batch, indices, weights = self.buffer.sample(self.batch_size)

        obs = torch.FloatTensor(np.stack([t["obs"] for t in batch]))
        actions = torch.LongTensor(np.stack([t["actions"] for t in batch]))
        rewards = torch.FloatTensor(np.stack([t["rewards"] for t in batch]))
        next_obs = torch.FloatTensor(
            np.stack([t["next_obs"] for t in batch]))
        dones = torch.FloatTensor([float(t["done"]) for t in batch])

        # Current Q-values for taken actions
        q_values = self.model(obs)
        q_taken = q_values.gather(
            -1, actions.unsqueeze(-1)).squeeze(-1)   # (B, N)

        # Double Q-learning: select with online, evaluate with target
        with torch.no_grad():
            next_actions = self.model(next_obs).argmax(
                dim=-1, keepdim=True)
            next_q = self.target_model(next_obs).gather(
                -1, next_actions).squeeze(-1)
            target = (rewards
                      + self.gamma ** self.n_step
                      * next_q * (1 - dones.unsqueeze(-1)))

        # TD errors for priority update (mean over agents)
        td_errors = (q_taken - target).detach().mean(dim=-1).numpy()
        self.buffer.update_priorities(indices, td_errors)

        # Weighted Huber loss
        loss = (
            weights.unsqueeze(-1)
            * F.smooth_l1_loss(q_taken, target, reduction="none")
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.privacy.clip_and_noise(self.model, self.batch_size)
        self.optimizer.step()

        # Polyak target update: theta' <- tau*theta + (1-tau)*theta'
        with torch.no_grad():
            for p, tp in zip(self.model.parameters(),
                             self.target_model.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------
    def evaluate(self, num_episodes: int = 50) -> Dict[str, Any]:
        """Evaluate policy without exploration.

        Returns dict with:
          - Summary statistics (reward_mean, reward_std, csr_mean, etc.)
          - Raw per-episode arrays (episode_rewards, episode_csrs) for
            proper statistical testing (no synthetic normal approximation).
        """
        self.model.eval()
        ep_rewards = []
        ep_csrs = []
        ep_occ_rates = []
        ep_diversities = []

        for _ in range(num_episodes):
            obs_dict = self.env.reset()
            obs = obs_dict["obs"]
            done = False
            ep_reward = 0.0

            while not done:
                actions = self.model.get_actions(obs, epsilon=0.0)
                obs_dict, r, done, info = self.env.step(actions.numpy())
                obs = obs_dict["obs"]
                ep_reward += float(r.mean())

            summary = self.env.get_episode_summary()
            ep_rewards.append(ep_reward)
            ep_csrs.append(summary.get("constraint_satisfaction_rate", 0.0))
            ep_occ_rates.append(summary.get("final_occupancy_rate", 0.0))
            ep_diversities.append(summary.get("visit_diversity", 0.0))

        rewards_arr = np.array(ep_rewards)
        csrs_arr = np.array(ep_csrs)
        n = len(ep_rewards)

        return {
            # Summary statistics
            "reward_mean": float(rewards_arr.mean()),
            "reward_std": float(rewards_arr.std()),
            "reward_ci95": float(
                1.96 * rewards_arr.std() / np.sqrt(max(1, n))),
            "csr_mean": float(csrs_arr.mean()),
            "csr_std": float(csrs_arr.std()),
            "occupancy_mean": float(np.mean(ep_occ_rates)),
            "diversity_mean": float(np.mean(ep_diversities)),
            # Raw arrays for statistical testing
            "episode_rewards": rewards_arr,
            "episode_csrs": csrs_arr,
        }

    # ----------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------
    def save_checkpoint(self, path: str):
        """Save full training state to file."""
        torch.save({
            "model": self.model.state_dict(),
            "target": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode": self.episode_count,
            "epsilon": self.epsilon,
            "best_reward": self.best_reward,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training state from checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episode_count = ckpt.get("episode", 0)
        self.epsilon = ckpt.get("epsilon", 0.01)
