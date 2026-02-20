"""
baselines.py - Strong MARL Baselines for Credible Comparison

Implements:
  - QMIX (Rashid et al., ICML 2018) - monotonic value decomposition
  - VDN (Sunehag et al., 2018) - additive value decomposition
  - Independent DQN - no coordination baseline
  - MAPPO (Yu et al., 2022) - multi-agent PPO with shared actor
  - Heuristic baselines: Random, Greedy, Distance-optimized

These are proper implementations to enable fair comparison,
not strawman baselines designed to be beaten.

Key fixes:
  - Factory passes distance_matrix and attractiveness properly
  - DistanceOptimizedBaseline works without per-step location tracking
  - All baselines expose consistent get_actions() interface
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


# ============================================================
# Shared Agent Network
# ============================================================
class AgentQNetwork(nn.Module):
    """Shared per-agent Q-network used by QMIX, VDN, IndependentDQN."""

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        return self.net(obs)


# ============================================================
# QMIX (Rashid et al., ICML 2018)
# ============================================================
class QMIXMixer(nn.Module):
    """QMIX mixing network with monotonicity constraint.

    Uses hypernetworks to generate mixing weights from global state.
    Monotonicity: dQ_tot/dQ_i >= 0 enforced via abs() on weights.
    """

    def __init__(self, num_agents: int, state_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: (batch, num_agents) individual Q-values
            state: (batch, state_dim) global state
        Returns:
            (batch,) joint Q-value Q_tot
        """
        B = agent_qs.shape[0]
        qs = agent_qs.view(B, 1, self.num_agents)

        w1 = torch.abs(self.hyper_w1(state)).view(
            B, self.num_agents, -1)
        b1 = self.hyper_b1(state).view(B, 1, -1)
        h = F.elu(torch.bmm(qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(B, -1, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = (torch.bmm(h, w2) + b2).squeeze(-1).squeeze(-1)
        return q_tot


class QMIXAgent(nn.Module):
    """QMIX: Monotonic value decomposition with hypernetwork mixer."""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int,
                 state_dim: int, hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_net = AgentQNetwork(obs_dim, action_dim, hidden_dim)
        self.mixer = QMIXMixer(num_agents, state_dim, hidden_dim // 2)

    def forward(self, obs: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        return self.agent_net(obs)

    def get_actions(self, obs: torch.Tensor, epsilon: float = 0.0,
                    **kwargs) -> torch.Tensor:
        squeeze = obs.dim() == 2
        with torch.no_grad():
            q = self.forward(obs)
        if squeeze:
            q = q.squeeze(0)
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, q.shape[:-1])
        return q.argmax(dim=-1)

    def mix(self, agent_qs: torch.Tensor,
            state: torch.Tensor) -> torch.Tensor:
        return self.mixer(agent_qs, state)

    def set_graph(self, *args, **kwargs):
        """No-op for interface compatibility."""
        pass


# ============================================================
# VDN (Sunehag et al., 2018)
# ============================================================
class VDNAgent(nn.Module):
    """Value Decomposition Network: Q_tot = sum(Q_i)."""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int,
                 hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_net = AgentQNetwork(obs_dim, action_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.agent_net(obs)

    def get_actions(self, obs: torch.Tensor, epsilon: float = 0.0,
                    **kwargs) -> torch.Tensor:
        squeeze = obs.dim() == 2
        with torch.no_grad():
            q = self.forward(obs)
        if squeeze:
            q = q.squeeze(0)
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, q.shape[:-1])
        return q.argmax(dim=-1)

    def mix(self, agent_qs: torch.Tensor,
            state: Optional[torch.Tensor] = None) -> torch.Tensor:
        return agent_qs.sum(dim=-1)

    def set_graph(self, *args, **kwargs):
        pass


# ============================================================
# Independent DQN
# ============================================================
class IndependentDQN(nn.Module):
    """Each agent learns independently -- no coordination."""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int,
                 hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            for _ in range(num_agents)
        ])

    def forward(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        B, N, D = obs.shape
        qs = []
        for i in range(min(N, self.num_agents)):
            qs.append(self.agent_nets[i](obs[:, i]))
        return torch.stack(qs, dim=1)

    def get_actions(self, obs: torch.Tensor, epsilon: float = 0.0,
                    **kwargs) -> torch.Tensor:
        squeeze = obs.dim() == 2
        with torch.no_grad():
            q = self.forward(obs)
        if squeeze:
            q = q.squeeze(0)
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, q.shape[:-1])
        return q.argmax(dim=-1)

    def set_graph(self, *args, **kwargs):
        pass


# ============================================================
# MAPPO (Yu et al., 2022)
# ============================================================
class MAPPOAgent(nn.Module):
    """Multi-Agent PPO with shared actor and centralized critic."""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int,
                 state_dim: Optional[int] = None,
                 hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        critic_input = state_dim or obs_dim * num_agents
        self.critic = nn.Sequential(
            nn.Linear(critic_input, hidden_dim * 2), nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        return self.actor(obs)

    def get_actions(self, obs: torch.Tensor, epsilon: float = 0.0,
                    **kwargs) -> torch.Tensor:
        squeeze = obs.dim() == 2
        with torch.no_grad():
            logits = self.forward(obs)
        if squeeze:
            logits = logits.squeeze(0)
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, logits.shape[:-1])
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(
            probs.view(-1, self.action_dim), 1
        ).view(logits.shape[:-1])

    def get_log_probs(self, obs: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def set_graph(self, *args, **kwargs):
        pass


# ============================================================
# Heuristic Baselines
# ============================================================
class RandomBaseline:
    """Uniform random action selection. Lower bound on performance."""

    def __init__(self, action_dim: int, num_agents: int, **kwargs):
        self.action_dim = action_dim
        self.num_agents = num_agents

    def get_actions(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        n = obs.shape[0] if obs.dim() == 2 else obs.shape[1]
        return torch.randint(0, self.action_dim, (n,))

    def parameters(self):
        return iter([torch.tensor(0.0)])

    def set_graph(self, *args, **kwargs):
        pass


class GreedyBaseline:
    """Always select highest-attractiveness POI (ignoring crowding)."""

    def __init__(self, action_dim: int, num_agents: int,
                 attractiveness: Optional[np.ndarray] = None, **kwargs):
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.attractiveness = attractiveness

    def get_actions(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        n = obs.shape[0] if obs.dim() == 2 else obs.shape[1]
        if self.attractiveness is not None:
            best = int(np.argmax(self.attractiveness))
            return torch.full((n,), best, dtype=torch.long)
        return torch.randint(0, self.action_dim, (n,))

    def parameters(self):
        return iter([torch.tensor(0.0)])

    def set_graph(self, *args, **kwargs):
        pass


class DistanceOptimizedBaseline:
    """Select POI based on occupancy-weighted distance heuristic.

    Uses observation structure: agent obs contains location one-hot
    (first num_pois dims) and occupancy norm (next num_pois dims).
    Selects POI minimizing distance * (1 + occupancy).
    """

    def __init__(self, action_dim: int, num_agents: int,
                 distance_matrix: Optional[np.ndarray] = None, **kwargs):
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.distance_matrix = distance_matrix

    def get_actions(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.squeeze(0)
        n = obs.shape[0]
        num_pois = self.action_dim

        acts = []
        for i in range(n):
            agent_obs = obs[i].numpy() if torch.is_tensor(obs[i]) else obs[i]
            # Extract location from one-hot (first num_pois dims)
            loc = int(np.argmax(agent_obs[:num_pois]))
            # Extract occupancy (next num_pois dims)
            occ = agent_obs[num_pois:2 * num_pois]

            if self.distance_matrix is not None:
                dists = self.distance_matrix[loc].copy()
                # Score = distance * (1 + occupancy), lower is better
                scores = dists * (1.0 + occ)
                scores[loc] = float("inf")   # avoid staying
                acts.append(int(np.argmin(scores)))
            else:
                acts.append(np.random.randint(0, self.action_dim))
        return torch.LongTensor(acts)

    def parameters(self):
        return iter([torch.tensor(0.0)])

    def set_graph(self, *args, **kwargs):
        pass


# ============================================================
# Factory
# ============================================================
BASELINE_REGISTRY = {
    "qmix": QMIXAgent,
    "vdn": VDNAgent,
    "independent_dqn": IndependentDQN,
    "mappo": MAPPOAgent,
    "random": RandomBaseline,
    "greedy": GreedyBaseline,
    "distance": DistanceOptimizedBaseline,
}


def create_baseline(name: str, obs_dim: int, action_dim: int,
                    num_agents: int, state_dim: Optional[int] = None,
                    **kwargs):
    """Factory function for baseline instantiation.

    Args:
        name: One of 'qmix', 'vdn', 'independent_dqn', 'mappo',
              'random', 'greedy', 'distance'.
        obs_dim: Per-agent observation dimension.
        action_dim: Number of actions.
        num_agents: Number of agents.
        state_dim: Global state dimension (for QMIX/MAPPO).
        **kwargs: Extra args passed to baseline (e.g., attractiveness,
                  distance_matrix).

    Returns:
        Baseline instance.
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline: {name}. "
            f"Available: {list(BASELINE_REGISTRY.keys())}")

    cls = BASELINE_REGISTRY[name]
    if name in ("random", "greedy", "distance"):
        return cls(action_dim=action_dim, num_agents=num_agents, **kwargs)
    return cls(
        obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents,
        state_dim=state_dim or obs_dim * num_agents, **kwargs,
    )
