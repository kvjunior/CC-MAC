"""
models.py - CC-MAC: Constraint-Certified Multi-Agent Coordination Architecture

Unified agent architecture integrating:
  1. GNN Spatial Encoder (Graph Attention Network over POI graph)
  2. Multi-Agent Attention Coordination (inter-agent communication)
  3. Dueling Q-Head with constraint-aware action masking
  4. Augmented Lagrangian constraint enforcement (in trainer.py)

Designed for CTDE: Centralized Training, Decentralized Execution.

Key fixes over prior version:
  - Removed unused num_transformer_layers parameter
  - Added set_graph() so GNN is actually used during training
  - Added use_dueling flag for proper ablation
  - GNN spatial features are stored as buffers (no gradient through graph)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


# ============================================================
# Graph Attention Layer
# ============================================================
class GATLayer(nn.Module):
    """Graph Attention layer with edge-conditioned attention and pre-norm residuals.

    Implements multi-head attention over graph edges with:
      - Edge bias (distance/travel-time-conditioned attention)
      - Sparse softmax for memory-efficient graph attention
      - Pre-norm residual connections + FFN
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.LongTensor,
                edge_bias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, D) node features
            edge_index: (2, E) source/destination indices
            edge_bias: (E, num_heads) learned edge attention biases
        Returns:
            (N, D) updated node features
        """
        N = h.shape[0]
        src, dst = edge_index

        q = self.q_proj(h).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(N, self.num_heads, self.head_dim)

        # Edge-conditioned attention scores
        attn = (q[dst] * k[src]).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = attn + edge_bias
        attn = self._sparse_softmax(attn, dst, N)

        # Message passing: weighted value aggregation
        msg = attn.unsqueeze(-1) * v[src]
        agg = torch.zeros(N, self.num_heads, self.head_dim, device=h.device)
        agg.scatter_add_(
            0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)
        agg = agg.reshape(N, -1)

        # Residual + FFN
        h = self.norm1(h + self.dropout(self.out_proj(agg)))
        h = self.norm2(h + self.ffn(h))
        return h

    @staticmethod
    def _sparse_softmax(scores: torch.Tensor, indices: torch.LongTensor,
                        num_nodes: int) -> torch.Tensor:
        """Numerically stable sparse softmax grouped by destination node."""
        max_vals = torch.full(
            (num_nodes, scores.shape[1]), -1e9, device=scores.device)
        max_vals.scatter_reduce_(
            0, indices.unsqueeze(-1).expand_as(scores), scores,
            reduce="amax", include_self=False,
        )
        scores = scores - max_vals[indices]
        exp_s = torch.exp(scores)
        sum_exp = torch.zeros(
            num_nodes, scores.shape[1], device=scores.device)
        sum_exp.scatter_add_(
            0, indices.unsqueeze(-1).expand_as(exp_s), exp_s)
        return exp_s / (sum_exp[indices] + 1e-8)


# ============================================================
# GNN Spatial Encoder
# ============================================================
class GNNSpatialEncoder(nn.Module):
    """Graph Attention Network for encoding POI spatial relationships.

    Processes static POI graph to produce per-POI spatial embeddings
    capturing neighborhood effects, distance-based attention, and
    category/capacity relationships.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (num_pois, node_dim) POI features
            edge_index: (2, E) edge indices
            edge_features: (E, edge_dim) edge features [distance, travel_time]
        Returns:
            (num_pois, hidden_dim) spatial embeddings
        """
        h = self.node_proj(node_features)
        edge_bias = self.edge_proj(edge_features)
        for layer in self.layers:
            h = layer(h, edge_index, edge_bias)
        return self.norm(h)


# ============================================================
# Multi-Agent Attention Coordination
# ============================================================
class MultiAgentAttention(nn.Module):
    """Inter-agent coordination via multi-head attention.

    Enables agents to share intended destination signals and coordinate
    to avoid simultaneous convergence on the same POI. Each agent
    attends to all other agents' hidden states, producing a
    coordination-aware representation.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, agent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_embeddings: (batch, num_agents, hidden_dim)
        Returns:
            (batch, num_agents, hidden_dim) coordinated embeddings
        """
        B, N, D = agent_embeddings.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(agent_embeddings).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(agent_embeddings).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(agent_embeddings).view(B, N, H, Dh).transpose(1, 2)

        attn = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh), dim=-1)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).reshape(B, N, D)

        return self.norm(
            agent_embeddings + self.dropout(self.out_proj(out)))


# ============================================================
# Q-Head Variants
# ============================================================
class DuelingQHead(nn.Module):
    """Dueling Q-network: Q(s,a) = V(s) + [A(s,a) - mean(A)].

    Separates state-value and action-advantage estimation,
    improving learning stability and sample efficiency.
    Supports constraint-aware masking of infeasible actions.
    """

    def __init__(self, hidden_dim: int, action_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, h: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        v = self.value_stream(h)
        a = self.advantage_stream(h)
        q = v + (a - a.mean(dim=-1, keepdim=True))
        if action_mask is not None:
            q = q.masked_fill(action_mask == 0, -1e9)
        return q


class StandardQHead(nn.Module):
    """Standard Q-network head (no dueling decomposition).

    Used as ablation baseline to measure dueling contribution.
    """

    def __init__(self, hidden_dim: int, action_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, h: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.net(h)
        if action_mask is not None:
            q = q.masked_fill(action_mask == 0, -1e9)
        return q


# ============================================================
# Main Agent
# ============================================================
class CCMACAgent(nn.Module):
    """Constraint-Certified Multi-Agent Coordination Agent.

    Integrates all components:
        obs -> ObsEncoder -> [+ GNN spatial context] -> Fusion
            -> MultiAgentAttention -> Q-Head -> actions

    Supports:
      - set_graph(): store POI graph for GNN spatial reasoning
      - Centralized training / decentralized execution (CTDE)
      - Monte Carlo dropout for epistemic uncertainty
      - Ablation flags: use_gnn, use_coordination, use_dueling

    Args:
        obs_dim: Per-agent observation dimension.
        action_dim: Number of actions (= num_pois).
        num_agents: Number of concurrent agents.
        hidden_dim: Hidden layer dimension throughout.
        num_gnn_layers: Number of GNN layers (0 to disable GNN).
        num_coord_heads: Number of multi-agent attention heads (0 to disable).
        gnn_node_dim: POI node feature dimension (default 8).
        gnn_edge_dim: Edge feature dimension (default 2).
        use_dueling: Whether to use dueling Q-head (False = standard).
        mc_dropout: Dropout rate (also used for MC uncertainty).
    """

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int,
                 hidden_dim: int = 128, num_gnn_layers: int = 2,
                 num_coord_heads: int = 4, gnn_node_dim: int = 8,
                 gnn_edge_dim: int = 2, use_dueling: bool = True,
                 mc_dropout: float = 0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.mc_dropout = mc_dropout
        self.use_gnn = num_gnn_layers > 0
        self.use_coordination = num_coord_heads > 0
        self.use_dueling = use_dueling

        # 1. Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(mc_dropout),
        )

        # 2. GNN spatial encoder (optional)
        if self.use_gnn:
            self.spatial_encoder = GNNSpatialEncoder(
                gnn_node_dim, gnn_edge_dim, hidden_dim,
                num_heads=4, num_layers=num_gnn_layers,
                dropout=mc_dropout,
            )
            # Fusion: concatenate obs encoding + spatial context -> hidden_dim
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.spatial_encoder = None
            self.fusion = None

        # 3. Multi-agent coordination (optional)
        if self.use_coordination:
            self.coordinator = MultiAgentAttention(
                hidden_dim, num_coord_heads, mc_dropout)
        else:
            self.coordinator = None

        # 4. Q-head (dueling or standard)
        if self.use_dueling:
            self.q_head = DuelingQHead(hidden_dim, action_dim, mc_dropout)
        else:
            self.q_head = StandardQHead(hidden_dim, action_dim, mc_dropout)

        # Graph buffers (set via set_graph, not part of gradient)
        self._poi_features = None
        self._edge_index = None
        self._edge_features = None

    def set_graph(self, poi_features: torch.Tensor,
                  edge_index: torch.LongTensor,
                  edge_features: torch.Tensor):
        """Store POI graph for GNN spatial encoding.

        Call this once after creating the agent (and again if the
        environment changes, e.g., cross-domain transfer).

        Args:
            poi_features: (num_pois, node_dim) POI node features.
            edge_index: (2, E) edge indices.
            edge_features: (E, edge_dim) edge features.
        """
        # Store as non-parameter buffers (move to same device as model)
        device = next(self.parameters()).device
        self._poi_features = poi_features.to(device)
        self._edge_index = edge_index.to(device)
        self._edge_features = edge_features.to(device)

    def forward(self, obs: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass producing Q-values.

        Args:
            obs: (batch, num_agents, obs_dim) or (num_agents, obs_dim)
            action_mask: optional (batch, num_agents, action_dim)
        Returns:
            (batch, num_agents, action_dim) Q-values
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        B, N, _ = obs.shape

        # 1. Encode per-agent observations
        h = self.obs_encoder(obs)   # (B, N, hidden_dim)

        # 2. Fuse with GNN spatial context (if available)
        if (self.use_gnn and self.spatial_encoder is not None
                and self._poi_features is not None):
            spatial = self.spatial_encoder(
                self._poi_features, self._edge_index, self._edge_features)
            # Mean-pool POI embeddings as global spatial context
            spatial_ctx = spatial.mean(dim=0)           # (hidden_dim,)
            spatial_ctx = spatial_ctx.unsqueeze(0).unsqueeze(0).expand(
                B, N, -1)                               # (B, N, hidden_dim)
            h = self.fusion(torch.cat([h, spatial_ctx], dim=-1))

        # 3. Multi-agent coordination
        if self.use_coordination and self.coordinator is not None:
            h = self.coordinator(h)

        # 4. Q-values
        return self.q_head(h, action_mask)

    def get_actions(self, obs: torch.Tensor, epsilon: float = 0.0,
                    action_mask: Optional[torch.Tensor] = None,
                    **kwargs) -> torch.Tensor:
        """Select actions with epsilon-greedy exploration.

        Args:
            obs: (num_agents, obs_dim) or (batch, num_agents, obs_dim)
            epsilon: exploration rate (0 = greedy)
        Returns:
            (num_agents,) or (batch, num_agents) action indices
        """
        squeeze = obs.dim() == 2
        with torch.no_grad():
            q = self.forward(obs, action_mask=action_mask)
        if squeeze:
            q = q.squeeze(0)
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, q.shape[:-1])
        return q.argmax(dim=-1)

    def get_uncertainty(self, obs: torch.Tensor,
                        num_samples: int = 10,
                        **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate epistemic uncertainty via MC dropout.

        Returns:
            (q_mean, q_std) each of shape matching forward output.
        """
        self.train()   # Enable dropout
        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                samples.append(self.forward(obs, **kwargs))
        self.eval()
        stacked = torch.stack(samples)
        return stacked.mean(0), stacked.std(0)


# ============================================================
# Centralized Critic (for training only)
# ============================================================
class CCMACCritic(nn.Module):
    """Centralized critic for joint value estimation during training.

    Takes global state + all agent actions -> scalar value.
    Not used during decentralized execution.
    """

    def __init__(self, global_state_dim: int, action_dim: int,
                 num_agents: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = global_state_dim + num_agents * action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor,
                actions_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([global_state, actions_onehot], dim=-1)
        return self.net(x)
