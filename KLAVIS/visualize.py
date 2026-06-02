# visualize.py — Manhattan (Scaled) Benchmark Dashboard
# Charts tailored specifically to the Manhattan (scaled) algorithm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Style ────────────────────────────────────────────────
BG          = "#0f1117"
PANEL       = "#1a1d27"
BORDER      = "#2a2d3a"
ACCENT      = "#4f8ef7"
ACCENT2     = "#a78bfa"
GREEN       = "#34d399"
RED         = "#f87171"
YELLOW      = "#fbbf24"
TEXT        = "#e2e8f0"
TEXT_DIM    = "#64748b"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT_DIM,
    "ytick.color":       TEXT_DIM,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  BORDER,
})


class BenchmarkVisualizer:
    def __init__(self, recorder, engine,
                 genuine_features, impostor_features=None, threshold=3.0,
                 benchmark_history=None, save_path=None):
        self.recorder         = recorder
        self.engine           = engine
        self.genuine_features = genuine_features
        self.impostor_features   = impostor_features or []
        self.threshold           = threshold
        self.benchmark_history   = benchmark_history or []
        self.save_path           = save_path  # if set, save PNG to this path

    # ─────────────────────────────────────────────────────
    # MAIN ENTRY
    # ─────────────────────────────────────────────────────
    def show_all(self):
        has_history = len(self.benchmark_history) >= 2
        rows = 3 if has_history else 2
        fig_h = 15 if has_history else 12

        fig = plt.figure(figsize=(18, fig_h))
        fig.suptitle(
            "KLAVIS — KEYSTROKE BIOMETRICS DASHBOARD",
            fontsize=14, fontweight="bold", color=ACCENT, y=0.98
        )

        gs = gridspec.GridSpec(rows, 3, figure=fig,
                               hspace=0.55, wspace=0.38,
                               left=0.06, right=0.97,
                               top=0.94, bottom=0.06)

        self._plot_score_distribution(fig.add_subplot(gs[0, 0]))
        self._plot_error_rates(fig.add_subplot(gs[0, 1]))
        self._plot_threshold_sweep(fig.add_subplot(gs[0, 2]))
        self._plot_mad_profile(fig.add_subplot(gs[1, 0]))
        self._plot_feature_contributions(fig.add_subplot(gs[1, 1]))
        self._plot_score_timeline(fig.add_subplot(gs[1, 2]))

        if has_history:
            # Third row: profile maturation story spans all 3 columns via 3 plots
            self._plot_maturation_eer(fig.add_subplot(gs[2, 0]))
            self._plot_maturation_far_frr(fig.add_subplot(gs[2, 1]))
            self._plot_maturation_samples(fig.add_subplot(gs[2, 2]))

        # Auto-save if a path was given
        if self.save_path:
            import os
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            fig.savefig(self.save_path, dpi=120, facecolor=BG,
                        bbox_inches="tight")
            print(f"📸 Saved chart to {self.save_path}")

        plt.show()

    # ─────────────────────────────────────────────────────
    # 1. SCORE DISTRIBUTION
    #    Shows genuine vs impostor score separation.
    # ─────────────────────────────────────────────────────
    def _plot_score_distribution(self, ax):
        genuine  = self.recorder.genuine_scores
        impostor = [s['score'] for s in self.recorder.impostor_scores]

        ax.set_title("Score Distribution\n(genuine vs impostor separation)",
                     fontweight="bold", fontsize=11)
        ax.set_xlabel("Scaled Manhattan Score")
        ax.set_ylabel("Count")

        # Choose shared bin range for clean overlay
        all_scores = genuine + impostor
        if all_scores:
            max_s = max(all_scores)
            bins  = np.linspace(0, max_s * 1.05, 20)

            if genuine:
                ax.hist(genuine, bins=bins, color=GREEN, alpha=0.7,
                        label=f"Genuine (n={len(genuine)})", edgecolor=BG)
            if impostor:
                ax.hist(impostor, bins=bins, color=RED, alpha=0.7,
                        label=f"Impostor (n={len(impostor)})", edgecolor=BG)

        ax.axvline(x=self.threshold, color=YELLOW, linestyle="--",
                   linewidth=1.5, label=f"Threshold = {self.threshold}")

        # Shade accept/reject regions
        ax.axvspan(0, self.threshold, alpha=0.05, color=GREEN)
        ax.axvspan(self.threshold, ax.get_xlim()[1] if ax.get_xlim()[1] > self.threshold else self.threshold + 1,
                   alpha=0.05, color=RED)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax.text(0.02, 0.97, "← ACCEPT | REJECT →",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=TEXT_DIM)

    # ─────────────────────────────────────────────────────
    # 2. ERROR RATES AT CURRENT THRESHOLD
    #    FAR, FRR, EER with interpretation banner.
    # ─────────────────────────────────────────────────────
    def _plot_error_rates(self, ax):
        far = self.recorder.compute_far(self.threshold) * 100
        frr = self.recorder.compute_frr(self.threshold) * 100
        eer_raw, eer_thr = self.recorder.compute_eer()
        eer = eer_raw * 100 if eer_raw is not None else 0.0

        ax.set_title(f"Error Rates @ threshold = {self.threshold}",
                     fontweight="bold", fontsize=11)

        metrics = ["FAR\n(impostors\naccepted)",
                   "FRR\n(genuine\nrejected)",
                   "EER\n(balanced\npoint)"]
        values  = [far, frr, eer]
        colors  = [RED, ACCENT, YELLOW]

        bars = ax.bar(metrics, values, color=colors, edgecolor=BG, width=0.55)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=TEXT)

        ax.set_ylabel("Error Rate (%)")
        ax.set_ylim(0, max(max(values) * 1.3, 20))
        ax.grid(True, axis="y", alpha=0.3)

        # Quality bands
        ax.axhline(y=10, color=GREEN,  linestyle="--", linewidth=1, alpha=0.4)
        ax.axhline(y=20, color=YELLOW, linestyle="--", linewidth=1, alpha=0.4)

        # Quality interpretation text
        if eer < 10:
            label, col = "✓ Good (EER < 10%)", GREEN
        elif eer < 20:
            label, col = "~ Acceptable", YELLOW
        else:
            label, col = "! Needs more data", RED
        ax.text(0.5, 0.95, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                fontweight="bold", color=col)

        if eer_thr is not None:
            ax.text(0.5, -0.22,
                    f"Optimal threshold (EER): {eer_thr:.3f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color=TEXT_DIM, style="italic")

    # ─────────────────────────────────────────────────────
    # 3. THRESHOLD SWEEP
    #    How FAR and FRR move as threshold changes.
    # ─────────────────────────────────────────────────────
    def _plot_threshold_sweep(self, ax):
        genuine  = self.recorder.genuine_scores
        impostor = [s['score'] for s in self.recorder.impostor_scores]

        ax.set_title("Threshold Sweep\n(FAR vs FRR tradeoff)",
                     fontweight="bold", fontsize=11)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate (%)")

        if not genuine or not impostor:
            ax.text(0.5, 0.5, "Need both genuine\n& impostor data",
                    transform=ax.transAxes, ha="center", va="center",
                    color=TEXT_DIM)
            return

        all_scores = genuine + impostor
        thresholds = np.linspace(0, max(all_scores) * 1.1, 200)
        fars = [self.recorder.compute_far(t) * 100 for t in thresholds]
        frrs = [self.recorder.compute_frr(t) * 100 for t in thresholds]

        ax.plot(thresholds, fars, color=RED,    linewidth=2, label="FAR")
        ax.plot(thresholds, frrs, color=ACCENT, linewidth=2, label="FRR")

        # Current threshold line
        ax.axvline(x=self.threshold, color=TEXT_DIM, linestyle=":",
                   linewidth=1, alpha=0.7,
                   label=f"Current ({self.threshold})")

        # EER crossing point
        eer_raw, eer_thr = self.recorder.compute_eer()
        if eer_raw is not None:
            ax.axvline(x=eer_thr, color=YELLOW, linestyle="--",
                       linewidth=1.5,
                       label=f"EER @ {eer_thr:.2f}")
            ax.scatter([eer_thr], [eer_raw * 100],
                       color=YELLOW, zorder=5, s=80,
                       edgecolor=BG, linewidth=1)

        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(all_scores) * 1.05)

    # ─────────────────────────────────────────────────────
    # 4. MAD PROFILE — THE KEY ALGORITHM INSIGHT
    #    Shows the scaling factors (MAD) per feature.
    #    Small MAD = consistent feature = strong biometric signal.
    #    Large MAD = variable feature = weak signal.
    # ─────────────────────────────────────────────────────
    def _plot_mad_profile(self, ax):
        ax.set_title("MAD Per Feature\n(consistency profile — smaller = more distinctive)",
                     fontweight="bold", fontsize=11)

        if self.engine.mad_vector is None:
            ax.text(0.5, 0.5, "No profile yet",
                    transform=ax.transAxes, ha="center", va="center",
                    color=TEXT_DIM)
            return

        bigrams = self.engine.bigram_order
        mad     = self.engine.mad_vector

        # Separate dwell and flight MADs (interleaved in the vector)
        dwell_mad  = mad[0::2]  # indices 0, 2, 4, ...
        flight_mad = mad[1::2]  # indices 1, 3, 5, ...

        x = np.arange(len(bigrams))
        w = 0.4

        ax.bar(x - w/2, dwell_mad  * 1000, w, label="Dwell MAD",
               color=ACCENT, alpha=0.8, edgecolor=BG)
        ax.bar(x + w/2, flight_mad * 1000, w, label="Flight MAD",
               color=ACCENT2, alpha=0.8, edgecolor=BG)

        ax.set_xticks(x)
        ax.set_xticklabels(bigrams, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("MAD (ms)")
        ax.set_xlabel("Bigram")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Highlight most consistent features (top 3 smallest MAD)
        total_mad = dwell_mad + flight_mad
        top_idx = np.argsort(total_mad)[:3]
        ax.text(0.02, 0.97,
                f"Most consistent: {', '.join(bigrams[i] for i in top_idx)}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=GREEN, fontweight="bold")

    # ─────────────────────────────────────────────────────
    # 5. FEATURE CONTRIBUTIONS
    #    Which bigrams contributed most to scoring?
    #    Averages scaled-distance per bigram across all genuine
    #    test attempts vs all impostor attempts.
    # ─────────────────────────────────────────────────────
    def _plot_feature_contributions(self, ax):
        ax.set_title("Score Contribution Per Bigram\n(which key pairs drove accept/reject?)",
                     fontweight="bold", fontsize=11)

        if (self.engine.mean_vector is None
                or not self.genuine_features
                or not self.impostor_features):
            ax.text(0.5, 0.5, "Need profile + tests",
                    transform=ax.transAxes, ha="center", va="center",
                    color=TEXT_DIM)
            return

        bigrams = self.engine.bigram_order
        mean    = self.engine.mean_vector
        mad     = self.engine.mad_vector

        def per_bigram_contrib(features):
            vec    = self.engine._to_vector(features)
            scaled = np.abs(vec - mean) / mad
            return (scaled[0::2] + scaled[1::2]) / 2

        # Real averages for both populations
        gen_contribs = np.mean(
            [per_bigram_contrib(f) for f in self.genuine_features], axis=0)
        imp_contribs = np.mean(
            [per_bigram_contrib(f) for f in self.impostor_features], axis=0)

        x = np.arange(len(bigrams))
        w = 0.4

        ax.bar(x - w/2, gen_contribs, w, label="Genuine (avg)",
               color=GREEN, alpha=0.8, edgecolor=BG)
        ax.bar(x + w/2, imp_contribs, w, label="Impostor (avg)",
               color=RED, alpha=0.8, edgecolor=BG)

        ax.set_xticks(x)
        ax.set_xticklabels(bigrams, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Scaled Distance")
        ax.set_xlabel("Bigram")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Show the bigrams with biggest genuine-vs-impostor gap
        # (these are the "most discriminating" features)
        gap     = imp_contribs - gen_contribs
        top_idx = np.argsort(gap)[-3:][::-1]
        ax.text(0.02, 0.97,
                f"Best discriminators: {', '.join(bigrams[i] for i in top_idx)}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=GREEN, fontweight="bold")

    # ─────────────────────────────────────────────────────
    # 6. SCORE TIMELINE
    #    Genuine and impostor scores in attempt order.
    # ─────────────────────────────────────────────────────
    def _plot_score_timeline(self, ax):
        ax.set_title("Score Timeline\n(attempt order — genuine vs impostor)",
                     fontweight="bold", fontsize=11)

        genuine  = self.recorder.genuine_scores
        impostor = [s['score'] for s in self.recorder.impostor_scores]

        if not genuine and not impostor:
            ax.text(0.5, 0.5, "No scores yet",
                    transform=ax.transAxes, ha="center", va="center",
                    color=TEXT_DIM)
            return

        if genuine:
            ax.plot(range(1, len(genuine) + 1), genuine,
                    color=GREEN, marker="o", linewidth=2, markersize=7,
                    label=f"Genuine (n={len(genuine)})")
        if impostor:
            ax.plot(range(1, len(impostor) + 1), impostor,
                    color=RED, marker="s", linewidth=2, markersize=7,
                    label=f"Impostor (n={len(impostor)})", linestyle="--")

        ax.axhline(y=self.threshold, color=YELLOW, linestyle="--",
                   linewidth=1.5, label=f"Threshold = {self.threshold}")
        ax.axhspan(0, self.threshold, alpha=0.05, color=GREEN)

        ax.set_xlabel("Attempt #")
        ax.set_ylabel("Scaled Manhattan Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────
    # 7. MATURATION — EER OVER BENCHMARK RUNS
    # ─────────────────────────────────────────────────────
    def _plot_maturation_eer(self, ax):
        ax.set_title("Profile Maturation — EER over time\n(lower = better)",
                     fontweight="bold", fontsize=11)

        runs = list(range(1, len(self.benchmark_history) + 1))
        eers = [h["eer"] * 100 if h.get("eer") is not None else None
                for h in self.benchmark_history]

        # Filter out None
        pts = [(r, e) for r, e in zip(runs, eers) if e is not None]
        if not pts:
            ax.text(0.5, 0.5, "no EER data", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_DIM)
            return

        rs, es = zip(*pts)
        ax.plot(rs, es, color=YELLOW, linewidth=2.5, marker="o",
                markersize=8, markerfacecolor=ACCENT,
                markeredgecolor=BG, markeredgewidth=2)

        for r, e in pts:
            ax.annotate(f"{e:.1f}%", xy=(r, e), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=TEXT)

        # Quality bands
        ax.axhspan(0, 10, alpha=0.1, color=GREEN)
        ax.axhspan(10, 20, alpha=0.1, color=YELLOW)
        ax.axhspan(20, 100, alpha=0.08, color=RED)

        ax.set_xlabel("Benchmark Run #")
        ax.set_ylabel("EER (%)")
        ax.set_xticks(rs)
        ax.set_ylim(0, max(max(es) * 1.3, 30))
        ax.grid(True, alpha=0.3)

        # Trend annotation
        if len(es) >= 2:
            delta = es[-1] - es[0]
            trend = "improving ✓" if delta < 0 else "regressing ⚠"
            tcolor = GREEN if delta < 0 else RED
            ax.text(0.02, 0.97, f"Trend: {trend} ({delta:+.1f}%)",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, color=tcolor, fontweight="bold")

    # ─────────────────────────────────────────────────────
    # 8. MATURATION — FAR/FRR TOGETHER
    # ─────────────────────────────────────────────────────
    def _plot_maturation_far_frr(self, ax):
        ax.set_title("Profile Maturation — FAR & FRR",
                     fontweight="bold", fontsize=11)

        runs = list(range(1, len(self.benchmark_history) + 1))
        fars = [h["far"] * 100 for h in self.benchmark_history]
        frrs = [h["frr"] * 100 for h in self.benchmark_history]

        ax.plot(runs, fars, color=RED,    linewidth=2, marker="o",
                markersize=7, label="FAR (impostors accepted)")
        ax.plot(runs, frrs, color=ACCENT, linewidth=2, marker="s",
                markersize=7, label="FRR (genuine rejected)")

        ax.set_xlabel("Benchmark Run #")
        ax.set_ylabel("Error Rate (%)")
        ax.set_xticks(runs)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(fars + frrs) * 1.3, 20))

    # ─────────────────────────────────────────────────────
    # 9. MATURATION — SAMPLE COUNT VS EER
    # ─────────────────────────────────────────────────────
    def _plot_maturation_samples(self, ax):
        ax.set_title("Sample Count vs EER\n(does more data help?)",
                     fontweight="bold", fontsize=11)

        samples = [h["sample_count"] for h in self.benchmark_history]
        eers    = [h["eer"] * 100 if h.get("eer") is not None else None
                   for h in self.benchmark_history]
        phases  = [h.get("phase", "growth") for h in self.benchmark_history]

        pts = [(s, e, p) for s, e, p in zip(samples, eers, phases) if e is not None]
        if not pts:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_DIM)
            return

        # Color by phase
        for s, e, p in pts:
            color = YELLOW if p == "growth" else GREEN
            ax.scatter(s, e, color=color, s=90, edgecolor=BG,
                       linewidth=1.5, zorder=3)

        # Connecting line
        ss, es, _ = zip(*pts)
        ax.plot(ss, es, color=TEXT_DIM, linewidth=1,
                linestyle="--", alpha=0.5, zorder=2)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=YELLOW,
                   markersize=10, label="Growth phase"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN,
                   markersize=10, label="Adaptive phase"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

        ax.set_xlabel("Profile Sample Count")
        ax.set_ylabel("EER (%)")
        ax.grid(True, alpha=0.3)