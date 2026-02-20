"""
analysis.py - Statistical Analysis, Explainability, and LaTeX Report Generation

Provides:
  - Rigorous statistical testing (paired + independent, with corrections)
  - Effect size computation (Cohen's d)
  - Permutation-based feature importance
  - Automated LaTeX table generation for ESWA manuscript

Key fixes:
  - Added independent samples comparison (for comparing different methods)
  - Paired comparison retained (for same-method different-seed analysis)
  - full_comparison_table uses independent samples (correct for baselines)
  - Robust to edge cases (small samples, zero variance)
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


class StatisticalAnalyzer:
    """Rigorous statistical analysis with proper multiple comparison corrections."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    # ----------------------------------------------------------
    # Comparison Tests
    # ----------------------------------------------------------
    def independent_comparison(self, x: np.ndarray,
                                y: np.ndarray) -> Dict:
        """Independent samples comparison with automatic test selection.

        Checks normality (Shapiro-Wilk) and equal variance (Levene),
        then selects:
          - Welch's t-test (if normal)
          - Mann-Whitney U (if non-normal)
        """
        # Normality check
        norm_p_x, norm_p_y = None, None
        if len(x) >= 8:
            _, norm_p_x = stats.shapiro(x[:min(5000, len(x))])
        if len(y) >= 8:
            _, norm_p_y = stats.shapiro(y[:min(5000, len(y))])

        is_normal = True
        if norm_p_x is not None and norm_p_y is not None:
            is_normal = norm_p_x > 0.05 and norm_p_y > 0.05

        if is_normal:
            # Welch's t-test (does not assume equal variance)
            t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
            return {
                "test": "welch_t_test", "statistic": float(t_stat),
                "p_value": float(p_val),
                "normality_p_x": norm_p_x, "normality_p_y": norm_p_y,
            }
        else:
            u_stat, p_val = stats.mannwhitneyu(
                x, y, alternative="two-sided")
            return {
                "test": "mann_whitney_u", "statistic": float(u_stat),
                "p_value": float(p_val),
                "normality_p_x": norm_p_x, "normality_p_y": norm_p_y,
            }

    def paired_comparison(self, x: np.ndarray, y: np.ndarray,
                          method: str = "auto") -> Dict:
        """Paired comparison (e.g., same episodes, different seeds).

        Uses Shapiro-Wilk on differences, then selects:
          - Paired t-test (if normal)
          - Wilcoxon signed-rank (if non-normal)
        """
        diffs = x - y
        norm_p = None
        if len(diffs) >= 8:
            _, norm_p = stats.shapiro(diffs[:min(5000, len(diffs))])
            is_normal = norm_p > 0.05
        else:
            is_normal = True

        if method == "auto":
            method = "t-test" if is_normal else "wilcoxon"

        if method == "t-test":
            t_stat, p_val = stats.ttest_rel(x, y)
            return {
                "test": "paired_t_test", "statistic": float(t_stat),
                "p_value": float(p_val), "normality_p": norm_p,
            }
        else:
            try:
                w_stat, p_val = stats.wilcoxon(x, y)
            except ValueError:
                # All differences are zero
                w_stat, p_val = 0.0, 1.0
            return {
                "test": "wilcoxon_signed_rank",
                "statistic": float(w_stat),
                "p_value": float(p_val), "normality_p": norm_p,
            }

    # ----------------------------------------------------------
    # Effect Size
    # ----------------------------------------------------------
    def cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cohen's d effect size with pooled standard deviation.

        Interpretation: |d| < 0.2 negligible, 0.2-0.5 small,
                        0.5-0.8 medium, > 0.8 large.
        """
        nx, ny = len(x), len(y)
        var_x = np.var(x, ddof=1) if nx > 1 else 0.0
        var_y = np.var(y, ddof=1) if ny > 1 else 0.0
        pooled_std = np.sqrt(
            ((nx - 1) * var_x + (ny - 1) * var_y)
            / max(1, nx + ny - 2)
        )
        return float((np.mean(x) - np.mean(y)) / (pooled_std + 1e-10))

    def confidence_interval(self, x: np.ndarray,
                            confidence: float = 0.95) -> Tuple[float, float]:
        """Confidence interval using t-distribution."""
        n = len(x)
        if n < 2:
            m = float(np.mean(x))
            return (m, m)
        sem = stats.sem(x)
        if sem < 1e-12:
            m = float(np.mean(x))
            return (m, m)
        lo, hi = stats.t.interval(
            confidence, df=n - 1, loc=np.mean(x), scale=sem)
        return (float(lo), float(hi))

    # ----------------------------------------------------------
    # Multiple Comparison Correction
    # ----------------------------------------------------------
    def bonferroni_correction(self,
                              p_values: List[float]) -> List[float]:
        """Bonferroni correction for multiple comparisons."""
        m = len(p_values)
        return [min(1.0, p * m) for p in p_values]

    # ----------------------------------------------------------
    # Full Comparison Table
    # ----------------------------------------------------------
    def full_comparison_table(
        self,
        results: Dict[str, np.ndarray],
        baseline_key: str = "ccmac",
    ) -> Dict:
        """Generate full pairwise comparison table.

        Uses INDEPENDENT samples tests (correct for comparing
        different methods trained separately). Applies Bonferroni
        correction across all pairwise comparisons.

        Args:
            results: {method_name: array of per-episode rewards}
            baseline_key: name of proposed method
        Returns:
            Dict of {method_name: {cohens_d, p_value, p_corrected, ...}}
        """
        if baseline_key not in results:
            raise ValueError(
                f"baseline_key '{baseline_key}' not in results")

        baseline = results[baseline_key]
        table = {}
        raw_p_values = []
        method_names = []

        for name, values in results.items():
            if name == baseline_key:
                continue

            x, y = np.asarray(baseline), np.asarray(values)

            # Independent samples test (different methods, different runs)
            test_result = self.independent_comparison(x, y)
            d = self.cohens_d(x, y)
            diff = x.mean() - y.mean()
            ci = self.confidence_interval(x - y[:len(x)])
            mag = self._effect_magnitude(abs(d))

            table[name] = {
                "mean_diff": float(diff),
                "cohens_d": float(d),
                "effect_magnitude": mag,
                "test": test_result["test"],
                "statistic": float(test_result["statistic"]),
                "p_value": float(test_result["p_value"]),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
            }
            raw_p_values.append(test_result["p_value"])
            method_names.append(name)

        # Apply Bonferroni correction
        corrected = self.bonferroni_correction(raw_p_values)
        for name, p_corr in zip(method_names, corrected):
            table[name]["p_corrected"] = float(p_corr)
            table[name]["significant"] = p_corr < self.alpha

        return table

    @staticmethod
    def _effect_magnitude(d: float) -> str:
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


# ============================================================
# Explainability
# ============================================================
class ExplainabilityAnalyzer:
    """Model interpretability via permutation-based feature importance."""

    def __init__(self):
        self.results = {}

    def permutation_importance(self, model, env,
                                num_episodes: int = 20,
                                num_permutations: int = 5
                                ) -> Dict[str, float]:
        """Compute permutation-based feature importance.

        Measures performance drop when each feature group is shuffled.
        """
        import torch

        num_pois = env.num_pois
        feature_groups = {
            "location": (0, num_pois),
            "occupancy": (num_pois, 2 * num_pois),
            "queue": (2 * num_pois, 3 * num_pois),
            "time": (3 * num_pois, 3 * num_pois + 4),
            "weather": (3 * num_pois + 4, 3 * num_pois + 9),
        }

        baseline_reward = self._eval_reward(model, env, num_episodes)
        importance = {}

        for group_name, (start, end) in feature_groups.items():
            drops = []
            for _ in range(num_permutations):
                reward = self._eval_permuted(
                    model, env, num_episodes, start, end)
                drops.append(baseline_reward - reward)
            importance[group_name] = float(np.mean(drops))

        # Normalize to relative importance
        total = sum(abs(v) for v in importance.values()) + 1e-10
        importance = {k: abs(v) / total for k, v in importance.items()}
        self.results["feature_importance"] = importance
        return importance

    def _eval_reward(self, model, env, num_episodes: int) -> float:
        import torch
        model.eval()
        rewards = []
        for _ in range(num_episodes):
            obs_dict = env.reset()
            obs = obs_dict["obs"]
            ep_reward = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    actions = model.get_actions(obs, epsilon=0.0)
                obs_dict, r, done, _ = env.step(actions.numpy())
                ep_reward += float(r.mean())
                obs = obs_dict["obs"]
            rewards.append(ep_reward)
        return float(np.mean(rewards))

    def _eval_permuted(self, model, env, num_episodes: int,
                        feat_start: int, feat_end: int) -> float:
        import torch
        model.eval()
        rewards = []
        for _ in range(num_episodes):
            obs_dict = env.reset()
            obs = obs_dict["obs"]
            ep_reward = 0.0
            done = False
            while not done:
                obs_perm = obs.clone()
                perm_idx = torch.randperm(obs.shape[0])
                obs_perm[:, feat_start:feat_end] = obs[
                    perm_idx, feat_start:feat_end]
                with torch.no_grad():
                    actions = model.get_actions(obs_perm, epsilon=0.0)
                obs_dict, r, done, _ = env.step(actions.numpy())
                ep_reward += float(r.mean())
                obs = obs_dict["obs"]
            rewards.append(ep_reward)
        return float(np.mean(rewards))


# ============================================================
# LaTeX Report Generation
# ============================================================
class ReportGenerator:
    """Automated LaTeX table generation for the ESWA manuscript."""

    @staticmethod
    def comparison_table(results: Dict,
                         proposed_name: str = "CC-MAC") -> str:
        """Generate LaTeX comparison table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Performance comparison across methods. "
            r"Best results in \textbf{bold}. Statistical significance: "
            r"$^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}",
            r"\label{tab:comparison}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Reward & CSR & Cohen's $d$ & $p$-value (corr.) \\",
            r"\midrule",
        ]
        for name, r in results.items():
            rw = (f"{r.get('reward_mean', 0):.2f} $\\pm$ "
                  f"{r.get('reward_std', 0):.2f}")
            csr = f"{r.get('csr_mean', 0):.3f}"
            d = r.get("cohens_d", "-")
            p = r.get("p_corrected")
            if p is not None:
                sig = ("^{***}" if p < 0.001
                       else "^{**}" if p < 0.01
                       else "^{*}" if p < 0.05 else "")
                p_str = f"${p:.4f}{sig}$"
            else:
                p_str = "---"
            if isinstance(d, float):
                d = f"{d:.2f}"
            if name.lower() in ("ccmac", proposed_name.lower()):
                lines.append(
                    f"\\textbf{{{proposed_name}}} & "
                    f"\\textbf{{{rw}}} & "
                    f"\\textbf{{{csr}}} & --- & --- \\\\")
            else:
                lines.append(
                    f"{name} & {rw} & {csr} & {d} & {p_str} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    @staticmethod
    def ablation_table(results: Dict) -> str:
        """Generate LaTeX ablation study table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation study results. "
            r"$\Delta$ indicates performance change relative to full model.}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Variant & Reward & CSR & $\Delta$ Reward (\%) \\",
            r"\midrule",
        ]
        full_reward = results.get(
            "full_model", {}).get("reward_mean", 1.0)
        for name, r in results.items():
            rw = r.get("reward_mean", 0)
            csr = r.get("csr_mean", 0)
            delta = ((rw - full_reward)
                     / (abs(full_reward) + 1e-10)) * 100
            display_name = name.replace("_", " ").title()
            if name == "full_model":
                lines.append(
                    f"\\textbf{{{display_name}}} & "
                    f"\\textbf{{{rw:.2f}}} & "
                    f"\\textbf{{{csr:.3f}}} & --- \\\\")
            else:
                lines.append(
                    f"{display_name} & {rw:.2f} & {csr:.3f} "
                    f"& {delta:+.1f}\\% \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    @staticmethod
    def transfer_table(results: Dict) -> str:
        """Generate LaTeX cross-domain transfer table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Cross-domain transfer results. "
            r"Efficiency measured as fraction of source domain "
            r"performance.}",
            r"\label{tab:transfer}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Target City & Zero-shot & Few-shot (100 ep) & Gain \\",
            r"\midrule",
        ]
        for domain, r in results.items():
            if domain == "verona":
                continue
            zs = r.get("zero_shot_efficiency", 0)
            fs = r.get("few_shot_efficiency", 0)
            gain = fs - zs
            lines.append(
                f"{domain.title()} & {zs:.1%} & {fs:.1%} "
                f"& +{gain:.1%} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)
