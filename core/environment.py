"""
environment.py - Tourism Environment with Realistic Dynamics and Hard Constraints

Implements a constrained multi-agent stochastic game for sustainable tourism.

State per agent:
  [agent_location_onehot(num_pois) | poi_occupancy_norm(num_pois) |
   poi_queue_norm(num_pois) | time_encoding(4) | weather_onehot(5) |
   agent_visit_history(num_pois) | agent_preferences(num_categories)]

Global state:
  [poi_occupancy_norm(num_pois) | poi_queue_norm(num_pois) |
   time_encoding(4) | weather_onehot(5)]

Actions: Discrete POI index
Constraints: capacity, CO2, diversity (entropy), queue time

Key fixes over prior version:
  - global_state no longer duplicates occ_norm
  - global_state_dim property for consistent dimension tracking
  - get_poi_node_features() for GNN integration
  - Explicit RNG via numpy Generator for reproducibility
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class POIConfig:
    """Point of Interest configuration with real-world attributes."""
    poi_id: int
    name: str
    category: str
    capacity: int
    service_rate: float       # visitors processed per hour
    base_attractiveness: float  # [0, 1]
    is_outdoor: bool
    latitude: float
    longitude: float


@dataclass
class EnvConfig:
    """Environment configuration with documented defaults."""
    num_pois: int = 18
    num_agents: int = 10
    episode_length_minutes: float = 480.0   # 8 hours
    dt_minutes: float = 15.0                # decision interval
    operating_hours: Tuple[float, float] = (8.0, 20.0)

    # Reward weights (must sum to 1.0)
    w_satisfaction: float = 0.3
    w_crowding: float = 0.3
    w_diversity: float = 0.2
    w_efficiency: float = 0.2

    # Constraint thresholds
    capacity_safety_factor: float = 1.0     # fraction of max capacity
    co2_budget_kg: float = 100.0            # per-step CO2 budget
    min_diversity_entropy: float = 1.5      # minimum Shannon entropy of visits
    max_queue_minutes: float = 30.0         # maximum queue length

    # Stochastic parameters
    arrival_noise_std: float = 0.1
    transport_delay_scale: float = 0.05


# ============================================================
# Verona POI Data (real coordinates and approximate capacities)
# ============================================================
def generate_verona_pois(num_pois: int = 18) -> List[POIConfig]:
    """Generate Verona POI configurations with actual GPS coordinates.

    Capacities and service rates are informed by public municipal data
    and tourism board reports; exact values are approximations.
    """
    templates = [
        # (name, category, capacity, service_rate, attractiveness, outdoor, lat, lon)
        ("Arena di Verona",           "monuments", 300, 120, 0.95, False, 45.4384, 10.9916),
        ("Casa di Giulietta",         "cultural",   80,  40, 0.90, False, 45.4421, 10.9988),
        ("Castelvecchio",             "museums",   150,  60, 0.85, False, 45.4397, 10.9858),
        ("Basilica di San Zeno",      "religious", 120,  50, 0.80, False, 45.4414, 10.9793),
        ("Torre dei Lamberti",        "monuments",  60,  25, 0.82, False, 45.4427, 10.9970),
        ("Piazza delle Erbe",         "cultural",  500, 200, 0.88,  True, 45.4432, 10.9977),
        ("Giardino Giusti",           "gardens",   100,  45, 0.75,  True, 45.4456, 11.0052),
        ("Duomo di Verona",           "religious", 130,  55, 0.78, False, 45.4491, 10.9969),
        ("Teatro Romano",             "monuments", 200,  80, 0.83,  True, 45.4471, 11.0001),
        ("Museo di Castelvecchio",    "museums",   100,  40, 0.77, False, 45.4398, 10.9860),
        ("Arche Scaligere",           "monuments",  70,  30, 0.72,  True, 45.4440, 10.9979),
        ("Chiesa Sant'Anastasia",     "religious", 110,  45, 0.74, False, 45.4450, 10.9994),
        ("Ponte Pietra",              "monuments", 400, 150, 0.86,  True, 45.4472, 10.9993),
        ("Museo Lapidario",           "museums",    60,  25, 0.65, False, 45.4380, 10.9929),
        ("Palazzo della Ragione",     "cultural",   90,  35, 0.70, False, 45.4430, 10.9975),
        ("Basilica di San Fermo",     "religious", 100,  40, 0.73, False, 45.4401, 10.9981),
        ("Museo Archeologico",        "museums",    80,  30, 0.68, False, 45.4474, 11.0005),
        ("Giardini Pubblici",         "gardens",   600, 250, 0.70,  True, 45.4350, 11.0030),
    ]

    pois = []
    for i in range(min(num_pois, len(templates))):
        name, cat, cap, sr, attr, outdoor, lat, lon = templates[i]
        pois.append(POIConfig(i, name, cat, cap, sr, attr, outdoor, lat, lon))
    return pois


# ============================================================
# Main Environment
# ============================================================
class TourismEnvironment:
    """Multi-agent constrained tourism environment.

    Implements realistic POI dynamics including:
      - Haversine-based travel distances
      - M/M/c queuing model for wait times
      - Weather Markov chain affecting outdoor attractiveness
      - Background tourist arrivals/departures (Poisson process)
      - Four hard constraints: capacity, CO2, diversity, queue time

    Attributes:
        obs_dim: Per-agent observation dimension.
        action_dim: Number of discrete actions (= num_pois).
        global_state_dim: Global state dimension for centralized critics.
    """

    WEATHER_STATES = ["sunny", "cloudy", "light_rain", "heavy_rain", "storm"]
    CATEGORIES = ["museums", "religious", "monuments", "cultural", "gardens"]
    NUM_CATEGORIES = 5

    def __init__(self, config: EnvConfig, poi_list: Optional[List[POIConfig]] = None):
        self.config = config
        self.pois = poi_list or generate_verona_pois(config.num_pois)
        self.num_pois = len(self.pois)
        self.num_agents = config.num_agents

        # Compute distance matrix from actual GPS coordinates
        self.distance_matrix = self._compute_distance_matrix()

        # === Observation dimensions (documented breakdown) ===
        self.obs_dim = (
            self.num_pois +          # agent location one-hot
            self.num_pois +          # POI occupancy (normalized by capacity)
            self.num_pois +          # POI queue length (normalized by capacity)
            4 +                      # time encoding: sin/cos(hour), sin/cos(progress)
            5 +                      # weather one-hot (5 states)
            self.num_pois +          # agent visit history (binary)
            self.NUM_CATEGORIES      # agent category preferences
        )
        self.action_dim = self.num_pois

        # === Global state dimensions (no duplication) ===
        self._global_state_dim = (
            self.num_pois +          # POI occupancy (normalized)
            self.num_pois +          # POI queue length (normalized)
            4 +                      # time encoding
            5                        # weather one-hot
        )

        # Weather Markov chain transition matrix
        self.weather_transition = np.array([
            [0.70, 0.20, 0.05, 0.03, 0.02],   # sunny
            [0.30, 0.40, 0.20, 0.08, 0.02],   # cloudy
            [0.10, 0.30, 0.30, 0.25, 0.05],   # light rain
            [0.05, 0.15, 0.30, 0.40, 0.10],   # heavy rain
            [0.02, 0.08, 0.20, 0.30, 0.40],   # storm
        ])
        # Multiplicative impact on attractiveness
        self.weather_outdoor_impact = np.array([1.0, 0.9, 0.6, 0.3, 0.1])
        self.weather_indoor_impact  = np.array([1.0, 1.0, 1.0, 0.95, 0.85])

        # Pre-compute POI attribute arrays for vectorized operations
        self.capacities     = np.array([p.capacity for p in self.pois], dtype=np.float32)
        self.service_rates  = np.array([p.service_rate for p in self.pois], dtype=np.float32)
        self.attractiveness = np.array([p.base_attractiveness for p in self.pois], dtype=np.float32)
        self.is_outdoor     = np.array([p.is_outdoor for p in self.pois], dtype=bool)
        self.poi_categories = np.array([self.CATEGORIES.index(p.category) for p in self.pois])

        self._rng = np.random.default_rng()
        self._reset_state()

    @property
    def global_state_dim(self) -> int:
        """Dimension of the global state vector."""
        return self._global_state_dim

    # ----------------------------------------------------------
    # Reset and Step
    # ----------------------------------------------------------
    def _reset_state(self):
        """Initialize/reset all episode state variables."""
        self.current_time = 0.0
        self.weather_state = 0
        self.occupancy = np.zeros(self.num_pois, dtype=np.float32)
        self.queue_lengths = np.zeros(self.num_pois, dtype=np.float32)
        self.agent_locations = np.zeros(self.num_agents, dtype=np.int64)
        self.agent_visit_history = np.zeros(
            (self.num_agents, self.num_pois), dtype=np.float32)
        self.agent_preferences = np.zeros(
            (self.num_agents, self.NUM_CATEGORIES), dtype=np.float32)
        self.episode_rewards = []
        self.episode_violations = []
        self.step_count = 0

    def reset(self, seed=None) -> Dict[str, torch.Tensor]:
        """Reset environment for a new episode.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Observation dict with keys 'obs', 'global_state', 'adj_matrix'.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        self.current_time = self.config.operating_hours[0]
        self.weather_state = int(self._rng.choice(3))  # start with fair weather

        # Initial occupancy: low random fill (0-20% of capacity)
        self.occupancy = (
            self._rng.uniform(0, 0.2, self.num_pois).astype(np.float32)
            * self.capacities
        )

        # Generate agent preferences via Dirichlet (diverse visitor profiles)
        self.agent_preferences = self._rng.dirichlet(
            np.ones(self.NUM_CATEGORIES) * 2.0, size=self.num_agents
        ).astype(np.float32)

        return self._get_observations()

    def step(self, actions) -> Tuple[Dict, np.ndarray, bool, Dict]:
        """Execute one environment step.

        Args:
            actions: np.ndarray of shape (num_agents,) with POI indices.

        Returns:
            (observations, rewards, done, info)
            - observations: dict with 'obs', 'global_state', 'adj_matrix'
            - rewards: np.ndarray of shape (num_agents,)
            - done: bool
            - info: dict with reward decomposition, constraint status, etc.
        """
        actions = np.clip(actions.astype(np.int64), 0, self.num_pois - 1)
        dt_hours = self.config.dt_minutes / 60.0

        # Advance time
        self.current_time += dt_hours
        self.step_count += 1

        # Weather transition (Markov chain)
        probs = self.weather_transition[self.weather_state]
        self.weather_state = int(self._rng.choice(5, p=probs))

        # Compute travel times (walking at ~4 km/h with random delay)
        travel_times = np.zeros(self.num_agents, dtype=np.float32)
        for i in range(self.num_agents):
            origin = self.agent_locations[i]
            dest = actions[i]
            if origin != dest:
                base_time = self.distance_matrix[origin, dest] / 4.0  # hours
                delay = self._rng.exponential(
                    self.config.transport_delay_scale) * base_time
                travel_times[i] = (base_time + delay) * 60  # minutes

        # Update agent locations
        prev_locations = self.agent_locations.copy()
        self.agent_locations = actions.copy()

        # Update POI occupancy using M/M/c queuing dynamics
        departures = np.bincount(
            prev_locations, minlength=self.num_pois).astype(np.float32)
        arrivals = np.bincount(
            actions, minlength=self.num_pois).astype(np.float32)

        # Background tourist flow (Poisson process)
        bg_arrivals = self._rng.poisson(
            self.service_rates * dt_hours * 0.5, self.num_pois
        ).astype(np.float32)
        bg_departures = self._rng.poisson(
            self.service_rates * dt_hours * 0.4, self.num_pois
        ).astype(np.float32)

        self.occupancy = np.maximum(
            0, self.occupancy - departures - bg_departures
               + arrivals + bg_arrivals)

        # Update queue lengths (M/M/c approximation)
        for j in range(self.num_pois):
            mu = self.service_rates[j] / 60.0   # service rate per minute
            lam = arrivals[j] / self.config.dt_minutes   # arrival rate per minute
            if lam > 0 and mu > 0:
                rho = lam / mu
                if rho < 0.95:
                    # M/M/1 expected queue length
                    self.queue_lengths[j] = rho / (1 - rho) * lam
                else:
                    # Saturated: queue grows
                    self.queue_lengths[j] = min(
                        self.queue_lengths[j] + arrivals[j],
                        self.capacities[j])
            else:
                # No arrivals: queue drains
                self.queue_lengths[j] = max(
                    0, self.queue_lengths[j] - mu * self.config.dt_minutes)

        # Update visit history (binary: visited or not)
        for i in range(self.num_agents):
            self.agent_visit_history[i, actions[i]] = 1.0

        # Compute decomposed rewards and constraint violations
        rewards, reward_info = self._compute_rewards(actions, travel_times)
        violations, constraint_info = self._evaluate_constraints(
            actions, travel_times)

        done = self.current_time >= self.config.operating_hours[1]

        info = {
            **reward_info, **constraint_info,
            "time": self.current_time,
            "step": self.step_count,
            "weather": self.WEATHER_STATES[self.weather_state],
            "mean_occupancy_rate": float(
                (self.occupancy / (self.capacities + 1e-8)).mean()),
        }

        self.episode_rewards.append(float(rewards.mean()))
        self.episode_violations.append(violations)

        return self._get_observations(), rewards, done, info

    # ----------------------------------------------------------
    # Observation Construction
    # ----------------------------------------------------------
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Construct structured observation dict.

        Returns dict with:
            'obs': (num_agents, obs_dim) per-agent observations
            'global_state': (global_state_dim,) shared state
            'adj_matrix': (num_pois, num_pois) distance-based adjacency
        """
        occ_norm = self.occupancy / (self.capacities + 1e-8)
        queue_norm = self.queue_lengths / (self.capacities + 1e-8)

        # Temporal encoding: sinusoidal hour + progress through operating window
        hour_frac = (
            (self.current_time - self.config.operating_hours[0])
            / (self.config.operating_hours[1] - self.config.operating_hours[0]
               + 1e-8)
        )
        time_enc = np.array([
            np.sin(2 * np.pi * self.current_time / 24),
            np.cos(2 * np.pi * self.current_time / 24),
            np.sin(2 * np.pi * hour_frac),
            np.cos(2 * np.pi * hour_frac),
        ], dtype=np.float32)

        # Weather one-hot
        weather_oh = np.zeros(5, dtype=np.float32)
        weather_oh[self.weather_state] = 1.0

        # Per-agent observations
        obs_list = []
        for i in range(self.num_agents):
            loc_oh = np.zeros(self.num_pois, dtype=np.float32)
            loc_oh[self.agent_locations[i]] = 1.0
            agent_obs = np.concatenate([
                loc_oh,                           # agent location
                occ_norm,                         # POI occupancy
                queue_norm,                       # POI queues
                time_enc,                         # temporal encoding
                weather_oh,                       # weather state
                self.agent_visit_history[i],      # visit history
                self.agent_preferences[i],        # preference embedding
            ])
            obs_list.append(agent_obs)

        # Global state (no duplication — used by centralized critics)
        global_state = np.concatenate([occ_norm, queue_norm, time_enc, weather_oh])
        assert global_state.shape[0] == self._global_state_dim, (
            f"Global state dim mismatch: {global_state.shape[0]} vs "
            f"{self._global_state_dim}")

        # Distance-based adjacency (Gaussian kernel, sigma=1km)
        adj = np.exp(-self.distance_matrix / 1.0)
        np.fill_diagonal(adj, 0)

        return {
            "obs": torch.FloatTensor(np.stack(obs_list)),
            "global_state": torch.FloatTensor(global_state),
            "adj_matrix": torch.FloatTensor(adj),
        }

    # ----------------------------------------------------------
    # Reward Computation (Decomposed)
    # ----------------------------------------------------------
    def _compute_rewards(self, actions, travel_times):
        """Compute per-agent decomposed rewards.

        Components:
          r_satisfaction: attractiveness * weather_impact * preference_match
          r_crowding: exponential penalty when occupancy > 50% capacity
          r_diversity: Shannon entropy of visited category distribution
          r_efficiency: travel time penalty (normalized by 60 min)

        Returns:
            (rewards: ndarray(num_agents,), info: dict)
        """
        r_sat = np.zeros(self.num_agents, dtype=np.float32)
        r_crowd = np.zeros(self.num_agents, dtype=np.float32)
        r_div = np.zeros(self.num_agents, dtype=np.float32)
        r_eff = np.zeros(self.num_agents, dtype=np.float32)

        w_impact = np.where(
            self.is_outdoor,
            self.weather_outdoor_impact[self.weather_state],
            self.weather_indoor_impact[self.weather_state],
        )

        for i in range(self.num_agents):
            poi = actions[i]
            cat_idx = self.poi_categories[poi]
            pref_match = self.agent_preferences[i, cat_idx]

            # Satisfaction: attractiveness * weather * preference
            r_sat[i] = (self.attractiveness[poi]
                        * w_impact[poi]
                        * (0.5 + 0.5 * pref_match))

            # Crowding: exponential penalty near capacity
            occ_rate = self.occupancy[poi] / (self.capacities[poi] + 1e-8)
            r_crowd[i] = (np.exp(3.0 * (occ_rate - 0.8))
                          if occ_rate > 0.5 else 0.0)

            # Diversity: entropy of visited category distribution
            visited_cats = self.poi_categories[
                self.agent_visit_history[i] > 0]
            if len(visited_cats) > 0:
                cat_counts = np.bincount(
                    visited_cats, minlength=self.NUM_CATEGORIES) + 1e-10
                cat_probs = cat_counts / cat_counts.sum()
                r_div[i] = (-np.sum(cat_probs * np.log(cat_probs))
                            / np.log(self.NUM_CATEGORIES + 1e-10))

            # Efficiency: penalize long travel
            r_eff[i] = max(0.0, 1.0 - travel_times[i] / 60.0)

        cfg = self.config
        rewards = (cfg.w_satisfaction * r_sat
                   - cfg.w_crowding * r_crowd
                   + cfg.w_diversity * r_div
                   + cfg.w_efficiency * r_eff)

        info = {
            "r_satisfaction": float(r_sat.mean()),
            "r_crowding": float(r_crowd.mean()),
            "r_diversity": float(r_div.mean()),
            "r_efficiency": float(r_eff.mean()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
        }
        return rewards, info

    # ----------------------------------------------------------
    # Constraint Evaluation
    # ----------------------------------------------------------
    def _evaluate_constraints(self, actions, travel_times):
        """Evaluate hard constraint satisfaction.

        Four constraints:
          capacity: sum of occupancy rates exceeding safety factor
          co2: total travel distance * emission factor - budget
          diversity: minimum entropy of action distribution
          queue: maximum queue length across POIs

        Returns:
            (violations: dict, info: dict)
            violations[k] > 0 means constraint k is violated.
        """
        violations = {}

        # 1. Capacity constraint: penalize over-capacity POIs
        occ_rates = self.occupancy / (self.capacities + 1e-8)
        violations["capacity"] = float(
            np.maximum(0, occ_rates - self.config.capacity_safety_factor).sum())

        # 2. CO2 constraint: total travel emissions
        total_dist = sum(
            self.distance_matrix[self.agent_locations[i], actions[i]]
            for i in range(self.num_agents)
        )
        co2_kg = total_dist * 0.12   # kg CO2 per km (walking + transit mix)
        violations["co2"] = float(
            max(0, co2_kg - self.config.co2_budget_kg))

        # 3. Diversity constraint: entropy of current action distribution
        visit_dist = np.bincount(
            actions, minlength=self.num_pois).astype(np.float32)
        visit_probs = (visit_dist + 1e-10) / (visit_dist.sum() + 1e-8)
        entropy = -np.sum(visit_probs * np.log(visit_probs))
        violations["diversity"] = float(
            max(0, self.config.min_diversity_entropy - entropy))

        # 4. Queue time constraint: worst-case queue
        violations["queue"] = float(
            max(0, self.queue_lengths.max() - self.config.max_queue_minutes))

        all_ok = all(v <= 0.01 for v in violations.values())
        info = {
            "constraint_satisfied": all_ok,
            "total_violation": sum(violations.values()),
            **{f"{k}_violation": v for k, v in violations.items()},
        }
        return violations, info

    # ----------------------------------------------------------
    # Graph / GNN Helpers
    # ----------------------------------------------------------
    def get_edge_index(self) -> torch.LongTensor:
        """Get edge indices for GNN (POIs within 2 km connected).

        Falls back to fully connected if no edges within threshold.
        Returns: (2, num_edges) LongTensor.
        """
        threshold_km = 2.0
        edges = []
        for i in range(self.num_pois):
            for j in range(self.num_pois):
                if i != j and self.distance_matrix[i, j] < threshold_km:
                    edges.append([i, j])
        # Fallback: fully connected if no edges within threshold
        if not edges:
            for i in range(self.num_pois):
                for j in range(self.num_pois):
                    if i != j:
                        edges.append([i, j])
        return torch.LongTensor(edges).t().contiguous()

    def get_edge_features(self) -> torch.FloatTensor:
        """Get edge features: [distance_km, travel_time_minutes].

        Returns: (num_edges, 2) FloatTensor.
        """
        edge_index = self.get_edge_index()
        feats = []
        for k in range(edge_index.shape[1]):
            i, j = edge_index[0, k].item(), edge_index[1, k].item()
            dist = self.distance_matrix[i, j]
            travel_time = dist / 4.0 * 60   # walking at 4 km/h
            feats.append([dist, travel_time])
        return torch.FloatTensor(feats)

    def get_poi_node_features(self) -> torch.FloatTensor:
        """Get POI node features for GNN.

        Feature vector per POI (8 dims):
          - category one-hot (5 dims)
          - normalized capacity (1 dim)
          - normalized service rate (1 dim)
          - base attractiveness (1 dim)

        Returns: (num_pois, 8) FloatTensor.
        """
        max_cap = self.capacities.max() + 1e-8
        max_sr = self.service_rates.max() + 1e-8
        features = []
        for j in range(self.num_pois):
            cat_oh = [0.0] * self.NUM_CATEGORIES
            cat_oh[self.poi_categories[j]] = 1.0
            features.append(cat_oh + [
                self.capacities[j] / max_cap,
                self.service_rates[j] / max_sr,
                self.attractiveness[j],
            ])
        return torch.FloatTensor(features)

    # ----------------------------------------------------------
    # Episode Summary
    # ----------------------------------------------------------
    def get_episode_summary(self) -> Dict[str, Any]:
        """Compute episode-level summary statistics.

        Returns dict with episode_reward, CSR, per-constraint violation
        counts, occupancy rate, and visit diversity.
        """
        if not self.episode_rewards:
            return {
                "episode_reward": 0.0,
                "mean_step_reward": 0.0,
                "constraint_satisfaction_rate": 0.0,
                "total_steps": 0,
            }

        violation_counts = defaultdict(int)
        for v in self.episode_violations:
            for k, val in v.items():
                if val > 0.01:
                    violation_counts[k] += 1

        total_steps = len(self.episode_violations)
        steps_with_any = sum(
            1 for v in self.episode_violations
            if any(val > 0.01 for val in v.values())
        )
        csr = 1.0 - steps_with_any / max(1, total_steps)

        return {
            "episode_reward": sum(self.episode_rewards),
            "mean_step_reward": float(np.mean(self.episode_rewards)),
            "constraint_satisfaction_rate": csr,
            "total_steps": total_steps,
            "capacity_violations": violation_counts.get("capacity", 0),
            "co2_violations": violation_counts.get("co2", 0),
            "diversity_violations": violation_counts.get("diversity", 0),
            "queue_violations": violation_counts.get("queue", 0),
            "final_occupancy_rate": float(
                (self.occupancy / (self.capacities + 1e-8)).mean()),
            "visit_diversity": float(
                self.agent_visit_history.sum(axis=1).mean()),
        }

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise Haversine distances between all POIs (km)."""
        n = len(self.pois)
        dist = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._haversine(
                    self.pois[i].latitude, self.pois[i].longitude,
                    self.pois[j].latitude, self.pois[j].longitude,
                )
                dist[i, j] = dist[j, i] = d
        return dist

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        """Haversine great-circle distance in km."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# Factory
# ============================================================
def make_env(config=None, num_pois=18, num_agents=10) -> TourismEnvironment:
    """Factory function for creating environments.

    Args:
        config: Optional EnvConfig. If None, uses defaults with given num_pois/agents.
        num_pois: Number of points of interest (used if config is None).
        num_agents: Number of agents (used if config is None).

    Returns:
        TourismEnvironment instance.
    """
    if config is None:
        config = EnvConfig(num_pois=num_pois, num_agents=num_agents)
    return TourismEnvironment(config)
