#!/usr/bin/env python3
"""
verify.py - Quick verification that all modules load and basic operations work.

Runs 5 tests:
  1. Environment creation and stepping
  2. CC-MAC model forward pass and action selection
  3. Baseline creation and action selection
  4. Trainer (2 episodes of training + evaluation)
  5. Statistical analysis utilities

Exit code 0 = all passed, 1 = failure.
"""
import sys
import numpy as np


def test_environment():
    """Test environment creation, reset, step, and summary."""
    print("=== Test 1: Environment ===")
    from core.environment import make_env

    env = make_env(num_pois=18, num_agents=10)
    print(f"  obs_dim={env.obs_dim}, action_dim={env.action_dim}, "
          f"global_state_dim={env.global_state_dim}")
    print(f"  distance_matrix: {env.distance_matrix.shape}")

    # Check dimension consistency
    assert env.global_state_dim == env.num_pois * 2 + 4 + 5, (
        f"global_state_dim mismatch: {env.global_state_dim}")

    obs = env.reset(seed=42)
    assert obs["obs"].shape == (env.num_agents, env.obs_dim), (
        f"obs shape: {obs['obs'].shape}")
    assert obs["global_state"].shape == (env.global_state_dim,), (
        f"global_state shape: {obs['global_state'].shape}")
    print(f"  obs={obs['obs'].shape}, "
          f"global={obs['global_state'].shape}")

    # Step a few times
    for i in range(3):
        actions = np.random.randint(0, env.action_dim, env.num_agents)
        obs, r, done, info = env.step(actions)
        assert r.shape == (env.num_agents,), f"reward shape: {r.shape}"

    # GNN features
    node_feats = env.get_poi_node_features()
    edge_index = env.get_edge_index()
    edge_feats = env.get_edge_features()
    assert node_feats.shape == (env.num_pois, 8), (
        f"node_feats: {node_feats.shape}")
    print(f"  GNN: nodes={node_feats.shape}, "
          f"edges={edge_index.shape}, "
          f"edge_feats={edge_feats.shape}")

    s = env.get_episode_summary()
    print(f"  CSR={s['constraint_satisfaction_rate']:.3f}")
    print("  PASSED\n")
    return env


def test_models(env):
    """Test CC-MAC model creation, forward pass, and actions."""
    print("=== Test 2: CC-MAC Model ===")
    from core.models import CCMACAgent

    model = CCMACAgent(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        num_agents=env.num_agents, hidden_dim=128,
        num_gnn_layers=2, num_coord_heads=4, use_dueling=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    # Set GNN graph
    model.set_graph(
        env.get_poi_node_features(),
        env.get_edge_index(),
        env.get_edge_features(),
    )
    print("  GNN graph set successfully")

    obs = env.reset(seed=42)
    q = model(obs["obs"])
    assert q.shape[-1] == env.action_dim, f"Q shape: {q.shape}"
    print(f"  Q: {q.shape}")

    a = model.get_actions(obs["obs"], epsilon=0.1)
    assert a.shape == (env.num_agents,), f"Actions shape: {a.shape}"
    print(f"  Actions: {a.numpy()}")

    # Test uncertainty
    q_mean, q_std = model.get_uncertainty(obs["obs"], num_samples=3)
    print(f"  Uncertainty: mean={q_mean.shape}, std={q_std.shape}")

    # Test standard Q-head variant (for ablation)
    model_no_dueling = CCMACAgent(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        num_agents=env.num_agents, hidden_dim=128,
        use_dueling=False,
    )
    q2 = model_no_dueling(obs["obs"])
    assert q2.shape == q.shape, "StandardQHead shape mismatch"
    print("  StandardQHead variant OK")

    print("  PASSED\n")
    return model


def test_baselines(env):
    """Test all baseline creation and action selection."""
    print("=== Test 3: Baselines ===")
    from baselines.baselines import create_baseline

    obs = env.reset(seed=42)
    baseline_names = [
        "qmix", "vdn", "independent_dqn", "mappo",
        "random", "greedy", "distance",
    ]

    for name in baseline_names:
        kwargs = {}
        if name == "greedy":
            kwargs["attractiveness"] = env.attractiveness
        if name == "distance":
            kwargs["distance_matrix"] = env.distance_matrix

        bl = create_baseline(
            name, env.obs_dim, env.action_dim, env.num_agents,
            state_dim=env.global_state_dim, **kwargs,
        )
        a = bl.get_actions(obs["obs"], epsilon=0.0)
        assert a.shape == (env.num_agents,), (
            f"{name} actions shape: {a.shape}")
        print(f"  {name}: actions={a.shape} OK")

    print("  PASSED\n")


def test_trainer(env, model):
    """Test trainer with 2 training episodes and evaluation."""
    print("=== Test 4: Trainer (2 episodes) ===")
    from core.trainer import CCMACTrainer

    cfg = {
        "gamma": 0.99, "tau": 0.005, "lr": 3e-4,
        "batch_size": 32, "n_step": 3, "grad_clip": 1.0,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.9,
        "buffer_size": 2000, "total_episodes": 10,
        "target_csr": 0.95, "lambda_lr": 0.01,
    }
    trainer = CCMACTrainer(model=model, env=env, config=cfg)

    for ep in range(2):
        m = trainer.train_episode()
        print(
            f"  Ep{ep}: r={m['episode_reward']:.3f} "
            f"CSR={m['constraint_satisfaction_rate']:.3f}")

    ev = trainer.evaluate(3)
    assert "episode_rewards" in ev, "evaluate() must return raw arrays"
    assert isinstance(ev["episode_rewards"], np.ndarray), (
        "episode_rewards must be ndarray")
    assert len(ev["episode_rewards"]) == 3, (
        f"Expected 3 episodes, got {len(ev['episode_rewards'])}")
    print(
        f"  Eval: {ev['reward_mean']:.3f}+/-{ev['reward_std']:.3f} "
        f"CSR={ev['csr_mean']:.3f}")
    print(f"  Raw episode rewards: {ev['episode_rewards']}")
    print("  PASSED\n")


def test_analysis():
    """Test statistical analysis utilities."""
    print("=== Test 5: Analysis ===")
    from utils.analysis import StatisticalAnalyzer, ReportGenerator

    a = StatisticalAnalyzer()
    rng = np.random.default_rng(42)
    x = rng.normal(5, 1, 50)
    y = rng.normal(4, 1, 50)

    # Effect size
    d = a.cohens_d(x, y)
    assert abs(d) > 0, "Cohen's d should be non-zero"
    print(f"  Cohen's d = {d:.3f}")

    # Confidence interval
    ci = a.confidence_interval(x)
    assert ci[0] < ci[1], "CI lower must be < upper"
    print(f"  CI = ({ci[0]:.3f}, {ci[1]:.3f})")

    # Independent comparison
    result = a.independent_comparison(x, y)
    assert "p_value" in result, "Missing p_value"
    print(f"  Independent test: {result['test']}, "
          f"p={result['p_value']:.6f}")

    # Full comparison table
    table = a.full_comparison_table(
        {"ccmac": x, "baseline": y}, baseline_key="ccmac")
    assert "baseline" in table, "Missing baseline in table"
    assert "p_corrected" in table["baseline"], "Missing p_corrected"
    print(f"  Comparison table OK: "
          f"p_corr={table['baseline']['p_corrected']:.6f}")

    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 50)
    print("CC-MAC Verification Suite")
    print("=" * 50)
    try:
        env = test_environment()
        model = test_models(env)
        test_baselines(env)
        test_trainer(env, model)
        test_analysis()
        print("=" * 50)
        print("ALL 5 TESTS PASSED")
        print("=" * 50)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
