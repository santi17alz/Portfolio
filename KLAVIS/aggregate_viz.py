# aggregate_viz.py — visualization for pooled benchmark data
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime


# Match main visualizer color scheme
BG       = "#0f1117"
PANEL    = "#1a1d27"
ACCENT   = "#4f8ef7"
ACCENT2  = "#a78bfa"
GREEN    = "#34d399"
RED      = "#f87171"
YELLOW   = "#fbbf24"
TEXT     = "#e2e8f0"
TEXT_DIM = "#64748b"


class AggregateVisualizer:
    """
    Renders a dashboard from BenchmarkAggregator output.
    Shows pooled distributions, ROC, threshold convergence, and cross-user comparison.
    """

    def __init__(self, single_data, cross_data=None, save_path=None):
        self.single = single_data
        self.cross  = cross_data
        self.save_path = save_path

        # Apply dark style
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor':  BG,
            'axes.facecolor':    PANEL,
            'axes.edgecolor':    TEXT_DIM,
            'axes.labelcolor':   TEXT,
            'xtick.color':       TEXT_DIM,
            'ytick.color':       TEXT_DIM,
            'text.color':        TEXT,
            'axes.titlecolor':   TEXT,
            'grid.color':        TEXT_DIM,
            'grid.alpha':        0.2,
        })

    def show_all(self):
        has_cross = self.cross is not None and self.cross['n_impostor_total'] > 0

        rows = 3 if has_cross else 2
        fig = plt.figure(figsize=(18, 5 * rows))
        fig.suptitle("KLAVIS — ALL-TIME AGGREGATE STATISTICS",
                     fontsize=14, fontweight="bold", color=ACCENT, y=0.99)

        gs = gridspec.GridSpec(rows, 3, figure=fig,
                               hspace=0.55, wspace=0.38,
                               left=0.06, right=0.97,
                               top=0.93, bottom=0.06)

        # Row 1: per-profile aggregate
        self._plot_pooled_distribution(fig.add_subplot(gs[0, 0]))
        self._plot_pooled_roc(fig.add_subplot(gs[0, 1]))
        self._plot_threshold_convergence(fig.add_subplot(gs[0, 2]))

        # Row 2: maturation insights
        self._plot_eer_vs_samples(fig.add_subplot(gs[1, 0]))
        self._plot_running_eer(fig.add_subplot(gs[1, 1]))
        self._plot_summary_panel(fig.add_subplot(gs[1, 2]))

        # Row 3: cross-user comparison (if data available)
        if has_cross:
            self._plot_cross_user_distribution(fig.add_subplot(gs[2, 0]))
            self._plot_cross_user_roc(fig.add_subplot(gs[2, 1]))
            self._plot_cross_vs_self(fig.add_subplot(gs[2, 2]))

        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            fig.savefig(self.save_path, dpi=120, facecolor=BG,
                        bbox_inches="tight")
            print(f"📸 Saved aggregate chart to {self.save_path}")

        plt.show()

    # ─────────────────────────────────────────────────────
    # ROW 1: PER-PROFILE
    # ─────────────────────────────────────────────────────
    def _plot_pooled_distribution(self, ax):
        ax.set_title(f"Pooled Score Distribution\n"
                     f"({self.single['n_genuine_total']} genuine + "
                     f"{self.single['n_impostor_total']} impostor across "
                     f"{self.single['n_runs']} runs)",
                     fontsize=10, fontweight='bold')

        gen = self.single['all_genuine']
        imp = self.single['all_impostor']

        if not gen and not imp:
            ax.text(0.5, 0.5, "no raw scores yet\n(re-run benchmarks to populate)",
                    transform=ax.transAxes, ha='center', va='center',
                    color=TEXT_DIM)
            return

        # Cap visualization at the 95th percentile of impostors so we don't
        # waste plot real estate on extreme outliers
        all_scores = gen + imp
        cap = np.percentile(all_scores, 95) if all_scores else 10
        bins = np.linspace(0, cap, 30)

        if gen:
            ax.hist(gen, bins=bins, color=GREEN, alpha=0.7,
                    label=f'Genuine (n={len(gen)})', edgecolor=BG)
        if imp:
            ax.hist(imp, bins=bins, color=RED, alpha=0.5,
                    label=f'Impostor (n={len(imp)})', edgecolor=BG)

        if self.single['pooled_eer_thresh'] is not None:
            ax.axvline(self.single['pooled_eer_thresh'], color=YELLOW,
                       linestyle='--', linewidth=2,
                       label=f"Pooled EER threshold = {self.single['pooled_eer_thresh']:.2f}")

        ax.set_xlabel("Scaled Manhattan Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, loc='upper right')

    def _plot_pooled_roc(self, ax):
        ax.set_title("Pooled ROC Curve\n(FAR vs FRR across all benchmarks)",
                     fontsize=10, fontweight='bold')

        far = self.single['far_curve']
        frr = self.single['frr_curve']

        if not far:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha='center', va='center', color=TEXT_DIM)
            return

        # ROC plot: FAR on x, 1-FRR on y (=TAR)
        far_arr = np.array(far) / 100.0
        tar_arr = 1.0 - np.array(frr) / 100.0
        ax.plot(far_arr, tar_arr, color=ACCENT, linewidth=2)
        ax.plot([0, 1], [0, 1], color=TEXT_DIM, linestyle='--',
                linewidth=1, label='Random (chance)')

        # Mark EER point
        if self.single['pooled_eer'] is not None:
            eer_pct = self.single['pooled_eer']
            ax.scatter([eer_pct], [1 - eer_pct], color=YELLOW, s=80,
                       zorder=5, label=f"EER = {eer_pct*100:.1f}%")

        ax.set_xlabel("False Accept Rate (FAR)")
        ax.set_ylabel("True Accept Rate (1 - FRR)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc='lower right')

    def _plot_threshold_convergence(self, ax):
        ax.set_title("Optimal Threshold Convergence\n"
                     "(does the EER threshold stabilize?)",
                     fontsize=10, fontweight='bold')

        thresholds = self.single['per_run_threshold']
        samples    = self.single['per_run_samples']
        if not thresholds or not samples:
            ax.text(0.5, 0.5, "need ≥2 runs",
                    transform=ax.transAxes, ha='center', va='center',
                    color=TEXT_DIM)
            return

        runs = list(range(1, len(thresholds) + 1))
        ax.plot(runs, thresholds, marker='o', color=ACCENT2,
                linewidth=2, markersize=8)

        # Pooled-threshold line as reference
        if self.single['pooled_eer_thresh'] is not None:
            ax.axhline(self.single['pooled_eer_thresh'], color=YELLOW,
                       linestyle='--', linewidth=2, alpha=0.7,
                       label=f"Pooled EER threshold = {self.single['pooled_eer_thresh']:.2f}")

        ax.set_xlabel("Benchmark Run #")
        ax.set_ylabel("EER Threshold (per-run)")
        ax.set_xticks(runs)
        ax.legend(fontsize=8)

    # ─────────────────────────────────────────────────────
    # ROW 2: MATURATION INSIGHTS
    # ─────────────────────────────────────────────────────
    def _plot_eer_vs_samples(self, ax):
        ax.set_title("EER vs Profile Sample Count\n(more data → lower error?)",
                     fontsize=10, fontweight='bold')

        eer     = self.single['per_run_eer']
        samples = self.single['per_run_samples']
        phases  = self.single['per_run_phase']

        pts = [(s, e * 100, p) for s, e, p in zip(samples, eer, phases)
               if e is not None]
        if not pts:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha='center', va='center', color=TEXT_DIM)
            return

        for s, e, p in pts:
            color = YELLOW if p == 'growth' else GREEN
            ax.scatter(s, e, color=color, s=80, edgecolor=BG,
                       linewidth=1.5, zorder=3)

        # Trend line (linear regression)
        ss = np.array([p[0] for p in pts])
        es = np.array([p[1] for p in pts])
        if len(pts) >= 2:
            slope, intercept = np.polyfit(ss, es, 1)
            x_line = np.linspace(ss.min(), ss.max(), 50)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=ACCENT, linestyle='--',
                    linewidth=1.5, alpha=0.7,
                    label=f"trend: {slope:+.2f} EER% per sample")
            ax.legend(fontsize=8, loc='upper right')

        ax.set_xlabel("Profile Sample Count")
        ax.set_ylabel("EER (%)")

    def _plot_running_eer(self, ax):
        ax.set_title("Cumulative Pooled EER\n"
                     "(EER computed using runs 1..N pooled together)",
                     fontsize=10, fontweight='bold')

        history = self.single['history']
        runs    = []
        cum_eer = []

        cum_gen = []
        cum_imp = []
        from aggregator import BenchmarkAggregator
        agg = BenchmarkAggregator()

        for i, run in enumerate(history, start=1):
            cum_gen.extend(run.get('genuine_scores', []))
            cum_imp.extend(run.get('impostor_scores', []))
            if cum_gen and cum_imp:
                _, _, _, eer, _ = agg._compute_pooled_curves(cum_gen, cum_imp)
                if eer is not None:
                    runs.append(i)
                    cum_eer.append(eer * 100)

        if not runs:
            ax.text(0.5, 0.5, "raw scores not yet captured\n(run new benchmarks)",
                    transform=ax.transAxes, ha='center', va='center',
                    color=TEXT_DIM)
            return

        ax.plot(runs, cum_eer, marker='o', color=GREEN,
                linewidth=2.5, markersize=8)

        # Quality bands
        ax.axhspan(0, 10, alpha=0.1, color=GREEN)
        ax.axhspan(10, 20, alpha=0.1, color=YELLOW)
        ax.axhspan(20, 100, alpha=0.08, color=RED)

        ax.set_xlabel("Runs Pooled (1..N)")
        ax.set_ylabel("Cumulative Pooled EER (%)")
        ax.set_xticks(runs)
        ax.set_ylim(0, max(max(cum_eer) * 1.3, 30))

    def _plot_summary_panel(self, ax):
        ax.axis('off')
        ax.set_title("Summary", fontsize=10, fontweight='bold')

        lines = [
            f"PROFILE: {self.single['username']}",
            f"",
            f"Benchmark runs: {self.single['n_runs']}",
            f"Genuine attempts (total): {self.single['n_genuine_total']}",
            f"Impostor attempts (total): {self.single['n_impostor_total']}",
            f"",
        ]
        if self.single['pooled_eer'] is not None:
            lines += [
                f"POOLED EER: {self.single['pooled_eer']*100:.2f}%",
                f"  @ threshold {self.single['pooled_eer_thresh']:.2f}",
            ]
        else:
            lines += [
                "POOLED EER: insufficient data",
                "  (older benchmarks lack raw scores)",
            ]

        if self.cross is not None and self.cross['n_impostor_total'] > 0:
            lines += [
                f"",
                f"CROSS-USER POOL:",
                f"  vs {len(self.cross['contributing_users'])} other users",
                f"  → {self.cross['n_impostor_total']} pooled impostor scores",
            ]
            if self.cross['pooled_eer'] is not None:
                lines += [
                    f"  EER: {self.cross['pooled_eer']*100:.2f}%",
                    f"  @ threshold {self.cross['pooled_eer_thresh']:.2f}",
                ]

        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                color=TEXT, family='monospace')

    # ─────────────────────────────────────────────────────
    # ROW 3: CROSS-USER
    # ─────────────────────────────────────────────────────
    def _plot_cross_user_distribution(self, ax):
        ax.set_title(f"Cross-User Score Distribution\n"
                     f"(your scores vs {len(self.cross['contributing_users'])} other users' typing)",
                     fontsize=10, fontweight='bold')

        gen = self.cross['all_genuine']
        imp = self.cross['all_impostor']

        if not gen and not imp:
            ax.text(0.5, 0.5, "no cross-user data",
                    transform=ax.transAxes, ha='center', va='center',
                    color=TEXT_DIM)
            return

        all_scores = gen + imp
        cap = np.percentile(all_scores, 95) if all_scores else 10
        bins = np.linspace(0, cap, 30)

        if gen:
            ax.hist(gen, bins=bins, color=GREEN, alpha=0.7,
                    label=f'You (n={len(gen)})', edgecolor=BG)
        if imp:
            ax.hist(imp, bins=bins, color=RED, alpha=0.5,
                    label=f'Other users (n={len(imp)})', edgecolor=BG)

        if self.cross['pooled_eer_thresh'] is not None:
            ax.axvline(self.cross['pooled_eer_thresh'], color=YELLOW,
                       linestyle='--', linewidth=2,
                       label=f"EER threshold = {self.cross['pooled_eer_thresh']:.2f}")

        ax.set_xlabel("Scaled Manhattan Score (against your profile)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, loc='upper right')

    def _plot_cross_user_roc(self, ax):
        ax.set_title("Cross-User ROC\n(true population-pooled evaluation)",
                     fontsize=10, fontweight='bold')

        far = self.cross['far_curve']
        frr = self.cross['frr_curve']

        if not far:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha='center', va='center', color=TEXT_DIM)
            return

        far_arr = np.array(far) / 100.0
        tar_arr = 1.0 - np.array(frr) / 100.0
        ax.plot(far_arr, tar_arr, color=ACCENT2, linewidth=2,
                label="Cross-user ROC")
        ax.plot([0, 1], [0, 1], color=TEXT_DIM, linestyle='--',
                linewidth=1, label='Random')

        if self.cross['pooled_eer'] is not None:
            eer_pct = self.cross['pooled_eer']
            ax.scatter([eer_pct], [1 - eer_pct], color=YELLOW, s=80,
                       zorder=5, label=f"EER = {eer_pct*100:.1f}%")

        ax.set_xlabel("FAR")
        ax.set_ylabel("1 - FRR")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc='lower right')

    def _plot_cross_vs_self(self, ax):
        ax.set_title("Self-Impostor vs Cross-User\n(which gives a better evaluation?)",
                     fontsize=10, fontweight='bold')

        labels = []
        eers   = []
        colors = []

        if self.single['pooled_eer'] is not None:
            labels.append("Self-pooled\n(your benchmark\nimpostors)")
            eers.append(self.single['pooled_eer'] * 100)
            colors.append(ACCENT)

        if self.cross['pooled_eer'] is not None:
            labels.append("Cross-user\n(other users\nas impostors)")
            eers.append(self.cross['pooled_eer'] * 100)
            colors.append(ACCENT2)

        if not eers:
            ax.text(0.5, 0.5, "no comparable data",
                    transform=ax.transAxes, ha='center', va='center',
                    color=TEXT_DIM)
            return

        bars = ax.bar(labels, eers, color=colors, edgecolor=BG, linewidth=2)
        for bar, val in zip(bars, eers):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha='center', fontweight='bold')

        ax.axhline(10, color=GREEN, linestyle=':', alpha=0.4,
                   label='10% benchmark target')
        ax.set_ylabel("EER (%)")
        ax.set_ylim(0, max(eers) * 1.4)
        ax.legend(fontsize=8, loc='upper right')