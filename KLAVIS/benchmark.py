# benchmark.py
import numpy as np

class BenchmarkRecorder:
    def __init__(self):
        self.genuine_scores = []   # scores from the real user
        self.impostor_scores = []  # scores from impostors

    # -------------------------
    # RECORDING
    # -------------------------
    def record_genuine(self, score):
        """Call this when the real user tests against their own profile."""
        self.genuine_scores.append(score)

    def record_impostor(self, score, label=None):
        """Call this when someone else tests against the real user's profile."""
        self.impostor_scores.append({
            'score': score,
            'label': label or f"impostor_{len(self.impostor_scores) + 1}"
        })

    # -------------------------
    # FAR & FRR AT A THRESHOLD
    # -------------------------
    def compute_far(self, threshold):
        """
        False Accept Rate at a given threshold.
        How many impostors were incorrectly accepted?
        """
        if not self.impostor_scores:
            return 0.0
        wrong_accepts = sum(
            1 for s in self.impostor_scores
            if s['score'] <= threshold
        )
        return wrong_accepts / len(self.impostor_scores)

    def compute_frr(self, threshold):
        """
        False Reject Rate at a given threshold.
        How many genuine attempts were incorrectly rejected?
        """
        if not self.genuine_scores:
            return 0.0
        wrong_rejects = sum(
            1 for s in self.genuine_scores
            if s > threshold
        )
        return wrong_rejects / len(self.genuine_scores)

    # -------------------------
    # EER
    # -------------------------
    def compute_eer(self):
        """
        Sweep thresholds across all recorded scores to find
        where FAR and FRR are closest — that's your EER.
        """
        if not self.genuine_scores or not self.impostor_scores:
            print("⚠️  Need both genuine and impostor scores to compute EER.")
            return None, None

        # Fine linear grid — stable EER estimates even at 3–5 samples.
        # Sweeping only exact observed scores causes large jumps at small N.
        all_scores = (
            self.genuine_scores +
            [s['score'] for s in self.impostor_scores]
        )
        thresholds = np.linspace(0, max(all_scores) * 1.1, 200).tolist()

        best_eer = float('inf')
        best_threshold = None

        for t in thresholds:
            far = self.compute_far(t)
            frr = self.compute_frr(t)
            # EER is where FAR and FRR are closest together
            diff = abs(far - frr)
            if diff < best_eer:
                best_eer = diff
                best_threshold = t
                best_far = far
                best_frr = frr

        eer_value = (best_far + best_frr) / 2  # average them at crossing point
        return round(eer_value, 4), round(best_threshold, 4)

    # -------------------------
    # FULL REPORT
    # -------------------------
    def report(self, threshold):
        """Print a full benchmark summary."""
        far = self.compute_far(threshold)
        frr = self.compute_frr(threshold)
        eer, eer_threshold = self.compute_eer()

        print("\n" + "="*45)
        print("         BENCHMARK REPORT")
        print("="*45)
        print(f"  Genuine attempts  : {len(self.genuine_scores)}")
        print(f"  Impostor attempts : {len(self.impostor_scores)}")
        print(f"  Current threshold : {threshold}")
        print("-"*45)
        print(f"  FAR  (impostors accepted) : {far*100:.1f}%")
        print(f"  FRR  (real user rejected) : {frr*100:.1f}%")
        print("-"*45)
        if eer is not None:
            print(f"  EER                       : {eer*100:.1f}%")
            print(f"  Optimal threshold (EER)   : {eer_threshold}")
        print("="*45)

        # Score breakdown
        print("\n  📈 Genuine scores  :", self.genuine_scores)
        print("  📉 Impostor scores :",
              [s['score'] for s in self.impostor_scores])