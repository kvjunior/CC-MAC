# CC-MAC: Constraint-Certified Multi-Agent Coordination for Sustainable Tourism

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Paper:** *Constraint-Certified Multi-Agent Coordination for Sustainable Tourism: An Augmented Lagrangian Approach with Graph-Based Spatial Reasoning*
>
> **Submitted to:** Expert Systems With Applications (Elsevier)

---

## Overview

CC-MAC is a multi-agent reinforcement learning system that coordinates concurrent tourist groups under hard operational constraints — capacity limits, carbon emission budgets, visit diversity requirements, and queue time bounds — using an augmented Lagrangian mechanism integrated into a graph-based coordination architecture.

**Core finding:** Constraint enforcement is achievable at negligible performance cost. Removing the augmented Lagrangian increases reward by only 0.5% while the constraint satisfaction rate (CSR) collapses by 26.0 percentage points (from 0.903 to 0.643).

### Key Results

| Metric | CC-MAC | Best Baseline (MAPPO) |
|:--|:--:|:--:|
| Mean reward | **13.51 ± 1.96** | 12.43 ± 2.18 |
| Constraint satisfaction rate | **92.1%** | 66.8% |
| Cohen's *d* (vs. MAPPO) | 0.52 (*p* < 0.003) | — |
| Inference latency | 2.84 ms (CPU) | — |
| Model size | 1.57 MB (FP32) | — |

All comparisons are statistically significant after Bonferroni correction (*m* = 7, α = 0.05).

---

## Architecture

CC-MAC comprises four modular components (410,719 parameters total):

```
Observation ─► Obs. Encoder ─► Fusion ─► Coordination ─► Dueling Q-Head ─► Actions
                                 ▲          Attention
                                 │
POI Graph ──► GNN Spatial ───────┘
              Encoder                    ┌──────────────────┐
                                         │  Aug. Lagrangian  │◄── Constraint
                                         │  (penalty loop)   │    Violations
                                         └──────────────────┘
```

| Component | Parameters | Share | Description |
|:--|--:|--:|:--|
| Observation encoder | 27,904 | 6.8% | 2-layer MLP, ReLU |
| GNN spatial encoder | 265,612 | 64.7% | 2 GAT layers, 4 heads, edge-conditioned |
| Fusion layer | 33,152 | 8.1% | Concatenation + linear projection |
| Coordination attention | 66,304 | 16.1% | 4-head cross-agent attention |
| Dueling Q-head | 17,747 | 4.3% | Value/advantage decomposition |

---

## Repository Structure

```
CC-MAC/
├── core/
│   ├── environment.py      # Tourism environment with M/M/c queuing, weather dynamics
│   ├── models.py           # CC-MAC agent (GNN + coordination + dueling Q-network)
│   └── trainer.py          # Training loop, prioritised replay, augmented Lagrangian
├── baselines/
│   └── baselines.py        # QMIX, VDN, MAPPO, Independent DQN, heuristics
├── utils/
│   └── analysis.py         # Statistical testing, effect sizes, LaTeX table generation
├── config.yaml             # Full experimental configuration
├── requirements.txt        # Python dependencies
├── run_experiments.py      # Complete 6-phase experimental pipeline
├── verify.py               # Quick verification (5 unit tests)
└── README.md
```

---

## Installation

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 2.0.0

### Setup

```bash
# Clone the repository
git clone https://github.com/[username]/CC-MAC.git
cd CC-MAC

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies (CPU-only, recommended for reproducibility)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Verify installation
python verify.py
```

Expected output from `verify.py`:

```
=== Test 1: Environment ===
  obs_dim=86, action_dim=18, global_state_dim=45
  PASSED
=== Test 2: CC-MAC Model ===
  PASSED
=== Test 3: Baselines ===
  PASSED
=== Test 4: Trainer ===
  PASSED
=== Test 5: Statistical Analysis ===
  PASSED
All 5 tests passed.
```

---

## Usage

### Full Experimental Pipeline

Reproduce all results reported in the paper (6 phases):

```bash
python run_experiments.py --mode full --episodes 5000
```

This executes:
1. **Phase 1** — Multi-seed training (5 seeds × 5,000 episodes)
2. **Phase 2** — Baseline comparison (7 baselines)
3. **Phase 3** — Statistical analysis (Bonferroni-corrected Welch's *t*-tests)
4. **Phase 4** — Ablation study (4 architectural variants)
5. **Phase 5** — Cross-domain transfer (Verona → Florence, Venice, Rome, Milan)
6. **Phase 6** — Production benchmarks (latency, throughput)

### Individual Modes

```bash
# Training only
python run_experiments.py --mode train --episodes 5000

# Ablation study only (2,000 episodes, seed 42)
python run_experiments.py --mode ablation --episodes 2000

# Quick test run
python run_experiments.py --mode full --episodes 100
```

### Configuration

All hyperparameters are centralised in `config.yaml`:

```yaml
# Key settings
model:
  hidden_dim: 128
  gnn:
    num_layers: 2
    num_heads: 4

training:
  gamma: 0.99
  learning_rate: 3.0e-4
  batch_size: 128
  constraints:
    method: "augmented_lagrangian"
    target_satisfaction_rate: 0.95
    lambda_learning_rate: 0.01
    rho_growth: 1.5

environment:
  num_pois: 18
  num_agents: 10
  reward_weights:
    satisfaction: 0.3
    crowding: 0.3
    diversity: 0.2
    efficiency: 0.2
```

---

## Experimental Design

### Environment

- **Domain:** 18 Verona POIs, 5 categories (monuments, museums, religious, cultural, gardens)
- **Agents:** 10 concurrent tourist groups
- **Episodes:** 32 time steps × 15-minute intervals = 8-hour operating window
- **Constraints:** Capacity (≤ rated), CO₂ (≤ 100 kg/step), diversity (≥ 1.5 nats), queue (≤ 30 min)

### Baselines

| Method | Type | Reference |
|:--|:--|:--|
| MAPPO | Multi-agent policy gradient | Yu et al. (2022) |
| QMIX | Monotonic value decomposition | Rashid et al. (2018) |
| VDN | Additive value decomposition | Sunehag et al. (2018) |
| Independent DQN | No coordination | — |
| Distance | Nearest-POI heuristic | — |
| Greedy | Highest-attractiveness heuristic | — |
| Random | Uniform random selection | — |

### Statistical Protocol

- **Normality:** Shapiro–Wilk test (*p* > 0.15 for all distributions)
- **Comparison:** Welch's *t*-test (independent samples)
- **Correction:** Bonferroni (*m* = 7, α<sub>corrected</sub> ≈ 0.00714)
- **Effect size:** Cohen's *d* with pooled standard deviation
- **Seeds:** 42, 1337, 2048, 3141, 9999

---

## Results Summary

### Ablation Study

| Variant | Reward | CSR | Δ Reward | Δ CSR |
|:--|:--:|:--:|:--:|:--:|
| **Full model** | **12.74** | **0.903** | — | — |
| No GNN | 12.03 | 0.886 | −5.6% | −1.7 pp |
| No coordination | 11.63 | 0.871 | −8.7% | −3.2 pp |
| No constraints | 12.81 | 0.643 | +0.5% | **−26.0 pp** |
| No dueling | 12.28 | 0.891 | −3.6% | −1.2 pp |

### Cross-Domain Transfer

| Target City | POIs | Zero-Shot Efficiency | Zero-Shot CSR | Few-Shot CSR |
|:--|:--:|:--:|:--:|:--:|
| Florence | 11 | 64.5% | 0.837 | 0.889 |
| Venice | 14 | 71.2% | 0.851 | 0.897 |
| Rome | 18 | 91.6% | 0.912 | 0.928 |
| Milan | 9 | 58.6% | 0.818 | 0.874 |

---

## Output Structure

After running the full pipeline, results are saved to `./results/`:

```
results/
├── checkpoints/          # Model weights per seed
├── metrics/              # JSON files with per-episode statistics
├── figures/              # PDF figures for the manuscript
│   ├── fig4_lagrangian_dynamics.pdf
│   ├── fig5_decomposition.pdf
│   ├── fig7_redistribution.pdf
│   └── fig8_temporal_weather.pdf
└── tables/               # LaTeX-formatted tables
```

---

## Reproducibility

Deterministic execution is enabled by default:

```yaml
reproducibility:
  deterministic: true
  torch_deterministic: true
```

Seed consistency across five runs: mean reward 13.51 ± 0.25 (CV = 1.9%), CSR 0.921 ± 0.008 (all seeds > 0.91).

---

## Hardware Requirements

| Configuration | Training (5,000 episodes) | Inference |
|:--|:--|:--|
| CPU only | ~2–4 hours | 2.84 ms / decision |
| GPU (optional) | ~30–60 minutes | < 1 ms / decision |
| RAM | ≥ 4 GB | ≥ 512 MB |
| Storage | ~200 MB (with checkpoints) | 1.57 MB (model only) |


---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
