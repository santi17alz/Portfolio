# aggregator.py — pools benchmark data across runs and users
# for proper statistical analysis with small per-benchmark sample sizes
import os
import json
from pathlib import Path
import numpy as np
from collections import defaultdict


_HERE = Path(__file__).parent
PROFILES_DIR = str(_HERE / "profiles")


class BenchmarkAggregator:
    """
    Pools raw scores across all benchmark runs (and optionally across users)
    to enable proper aggregate statistics on small individual samples.
    """

    def __init__(self, profiles_dir=PROFILES_DIR):
        self.profiles_dir = profiles_dir

    # ─────────────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────────────
    def load_all_profiles(self):
        """Load every profile JSON in the profiles dir."""
        profiles = {}
        if not os.path.exists(self.profiles_dir):
            return profiles
        for fname in os.listdir(self.profiles_dir):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(self.profiles_dir, fname)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                username = fname.replace('.json', '')
                profiles[username] = data
            except (json.JSONDecodeError, IOError):
                continue  # skip malformed files
        return profiles

    # ─────────────────────────────────────────────────────
    # SINGLE-USER AGGREGATION
    # ─────────────────────────────────────────────────────
    def aggregate_single_user(self, username):
        """
        Pool all scores from all benchmark runs for one user.
        Returns dict with aggregated stats and raw pooled data.
        """
        profiles = self.load_all_profiles()
        if username not in profiles:
            return None
        history = profiles[username].get('benchmark_history', [])
        if not history:
            return None

        all_genuine = []
        all_impostor = []
        per_run_eer = []
        per_run_threshold = []
        per_run_samples = []
        per_run_phase = []

        for run in history:
            # Use new-format raw scores if available, else skip raw aggregation for that run
            all_genuine.extend(run.get('genuine_scores', []))
            all_impostor.extend(run.get('impostor_scores', []))
            if run.get('eer') is not None:
                per_run_eer.append(run['eer'])
                per_run_threshold.append(run.get('eer_threshold', 0))
            per_run_samples.append(run.get('sample_count', 0))
            per_run_phase.append(run.get('phase', 'unknown'))

        # Compute pooled error rates as a function of threshold
        far_curve, frr_curve, threshold_grid, pooled_eer, pooled_eer_thresh = \
            self._compute_pooled_curves(all_genuine, all_impostor)

        return {
            'username':          username,
            'n_runs':            len(history),
            'n_genuine_total':   len(all_genuine),
            'n_impostor_total':  len(all_impostor),
            'all_genuine':       all_genuine,
            'all_impostor':      all_impostor,
            'per_run_eer':       per_run_eer,
            'per_run_threshold': per_run_threshold,
            'per_run_samples':   per_run_samples,
            'per_run_phase':     per_run_phase,
            'history':           history,
            'pooled_eer':        pooled_eer,
            'pooled_eer_thresh': pooled_eer_thresh,
            'threshold_grid':    threshold_grid,
            'far_curve':         far_curve,
            'frr_curve':         frr_curve,
        }

    # ─────────────────────────────────────────────────────
    # CROSS-USER POOLING
    # ─────────────────────────────────────────────────────
    def aggregate_cross_user(self, target_username, engine_class):
        """
        Build a virtual benchmark for target_username by pooling impostor
        data from OTHER users' genuine and impostor attempts.

        This requires re-scoring those features against target's profile
        (since scores are profile-specific).

        Parameters
        ----------
        target_username : whose profile we're evaluating
        engine_class    : the engine class (e.g. ManhattanScaledEngine) to instantiate
                          and load with the target's profile
        """
        profiles = self.load_all_profiles()
        if target_username not in profiles:
            return None

        target_profile = profiles[target_username]

        # Load target's engine
        engine = engine_class()
        engine.load_from_profile(target_profile)

        # Collect target's own genuine scores from their benchmark history
        target_genuine = []
        for run in target_profile.get('benchmark_history', []):
            target_genuine.extend(run.get('genuine_scores', []))

        # Collect impostor scores from other users
        # - their bench_impostor_features were typed by impostors → impostors for everyone
        # - their genuine_features were typed by THEM → impostors for target
        pooled_impostor_features = []
        contributing_users = []
        for other_user, other_profile in profiles.items():
            if other_user == target_username:
                continue
            count_before = len(pooled_impostor_features)
            for run in other_profile.get('benchmark_history', []):
                pooled_impostor_features.extend(run.get('genuine_features',  []))
                pooled_impostor_features.extend(run.get('impostor_features', []))
            if len(pooled_impostor_features) > count_before:
                contributing_users.append(other_user)

        # Re-score every pooled impostor feature against target's profile
        pooled_impostor_scores = []
        for feat in pooled_impostor_features:
            try:
                score = engine.score(feat)
                if score is not None and not np.isnan(score) and np.isfinite(score):
                    pooled_impostor_scores.append(float(score))
            except (KeyError, ValueError, ZeroDivisionError):
                continue

        # Compute pooled curves
        far_curve, frr_curve, threshold_grid, pooled_eer, pooled_eer_thresh = \
            self._compute_pooled_curves(target_genuine, pooled_impostor_scores)

        return {
            'target_username':       target_username,
            'contributing_users':    contributing_users,
            'n_genuine_total':       len(target_genuine),
            'n_impostor_total':      len(pooled_impostor_scores),
            'all_genuine':           target_genuine,
            'all_impostor':          pooled_impostor_scores,
            'pooled_eer':            pooled_eer,
            'pooled_eer_thresh':     pooled_eer_thresh,
            'threshold_grid':        threshold_grid,
            'far_curve':             far_curve,
            'frr_curve':             frr_curve,
        }

    # ─────────────────────────────────────────────────────
    # POOLED ROC / EER COMPUTATION
    # ─────────────────────────────────────────────────────
    def _compute_pooled_curves(self, genuine_scores, impostor_scores):
        """
        For a sweep of thresholds, compute FAR and FRR.
        Returns (far_curve, frr_curve, threshold_grid, eer, eer_threshold).
        """
        if not genuine_scores or not impostor_scores:
            return [], [], [], None, None

        gen_arr = np.array(genuine_scores)
        imp_arr = np.array(impostor_scores)

        # Build threshold grid spanning all observed scores
        all_scores = np.concatenate([gen_arr, imp_arr])
        s_min      = float(np.min(all_scores))
        s_max      = float(np.max(all_scores))
        # Add a little padding so curves go to 0 and 100 at extremes
        threshold_grid = np.linspace(max(0.0, s_min - 0.1),
                                     s_max + 0.1, 200)

        far_curve = []
        frr_curve = []
        for t in threshold_grid:
            far = np.mean(imp_arr <= t) * 100.0  # impostor accepted
            frr = np.mean(gen_arr  >  t) * 100.0  # genuine rejected
            far_curve.append(far)
            frr_curve.append(frr)

        far_arr = np.array(far_curve)
        frr_arr = np.array(frr_curve)
        diff    = np.abs(far_arr - frr_arr)
        idx     = int(np.argmin(diff))
        eer     = float((far_arr[idx] + frr_arr[idx]) / 2.0)
        eer_t   = float(threshold_grid[idx])

        return far_curve, frr_curve, threshold_grid.tolist(), eer / 100.0, eer_t

    # ─────────────────────────────────────────────────────
    # USEFUL SUMMARIES
    # ─────────────────────────────────────────────────────
    def summary_text(self, single_data, cross_data=None):
        """Build a short text summary for display in the UI."""
        lines = []
        if single_data is None:
            return "No benchmark data yet."

        lines.append(f"PER-PROFILE AGGREGATE")
        lines.append(f"  • {single_data['n_runs']} benchmark runs")
        lines.append(f"  • {single_data['n_genuine_total']} total genuine attempts")
        lines.append(f"  • {single_data['n_impostor_total']} total impostor attempts")
        if single_data['pooled_eer'] is not None:
            lines.append(f"  • Pooled EER: {single_data['pooled_eer']*100:.1f}% "
                         f"@ threshold {single_data['pooled_eer_thresh']:.2f}")

        if cross_data is not None:
            lines.append("")
            lines.append(f"CROSS-USER POOLED (vs {len(cross_data['contributing_users'])} others)")
            lines.append(f"  • {cross_data['n_genuine_total']} genuine (you)")
            lines.append(f"  • {cross_data['n_impostor_total']} impostor (pooled from others)")
            if cross_data['pooled_eer'] is not None:
                lines.append(f"  • Cross-user EER: {cross_data['pooled_eer']*100:.1f}% "
                             f"@ threshold {cross_data['pooled_eer_thresh']:.2f}")
        return "\n".join(lines)