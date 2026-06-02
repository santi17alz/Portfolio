# engine.py — Manhattan (Scaled) Detector
# Based on: Araújo et al. (2004) — top performer in Killourhy & Maxion benchmark
# EER of ~9.6% on CMU dataset with 51 users
#
# Why this algorithm over SVM:
#   - Works reliably with small enrollment datasets (5-20 samples)
#   - Per-feature scaling handles dwell vs flight timing variability
#   - Transparent, interpretable, no hyperparameter tuning needed
#
# Retains Bloom filter + IQR outlier filtering as enhancement layers.

import logging
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# BLOOM FILTER
# ─────────────────────────────────────────────────────────
class BloomFilter:
    """Fast pre-check: reject attempts with bigrams never seen in enrollment."""
    def __init__(self, size=1024, num_hashes=4):
        self.size       = size
        self.num_hashes = num_hashes
        self.bit_array  = [0] * size

    def _hashes(self, item):
        return [
            int(hashlib.sha256(f"{i}:{item}".encode()).hexdigest(), 16) % self.size
            for i in range(self.num_hashes)
        ]

    def add(self, item):
        for pos in self._hashes(item):
            self.bit_array[pos] = 1

    def contains(self, item):
        return all(self.bit_array[pos] == 1 for pos in self._hashes(item))

    def coverage(self):
        return sum(self.bit_array) / self.size


# ─────────────────────────────────────────────────────────
# MANHATTAN (SCALED) ENGINE
# ─────────────────────────────────────────────────────────
class ManhattanScaledEngine:
    def __init__(self, threshold=3.0, outlier_iqr_factor=1.5):
        """
        threshold          : scaled-distance cutoff for accept/reject.
                             Tune based on benchmark results.
        outlier_iqr_factor : how aggressively to drop outlier enrollment samples.
        """
        self.threshold          = threshold
        self.outlier_iqr_factor = outlier_iqr_factor

        # Profile state built during enrollment
        self.bigram_order = []
        self.mean_vector  = None   # per-feature averages
        self.mad_vector   = None   # per-feature mean absolute deviations
        self.bloom        = BloomFilter()

    # ─────────────────────────────────────────────────────
    # FEATURE VECTOR BUILDER
    # ─────────────────────────────────────────────────────
    def _to_vector(self, features):
        """Convert bigram list into a flat numpy vector of [dwell,flight,...]"""
        vec = []
        feature_map = {f['bigram']: f for f in features}
        for bigram in self.bigram_order:
            if bigram in feature_map:
                vec.append(feature_map[bigram]['dwell'])
                vec.append(feature_map[bigram]['flight'])
            else:
                vec.append(0.0)
                vec.append(0.0)
        return np.array(vec)

    # ─────────────────────────────────────────────────────
    # OUTLIER FILTERING (IQR on total timing)
    # ─────────────────────────────────────────────────────
    def _filter_outliers(self, all_features):
        if len(all_features) < 6:
            return all_features

        totals = np.array([
            sum(f['dwell'] + f['flight'] for f in attempt)
            for attempt in all_features
        ])
        Q1, Q3 = np.percentile(totals, [25, 75])
        IQR    = Q3 - Q1
        lower  = Q1 - self.outlier_iqr_factor * IQR
        upper  = Q3 + self.outlier_iqr_factor * IQR

        filtered = [a for a, t in zip(all_features, totals) if lower <= t <= upper]
        dropped  = len(all_features) - len(filtered)

        logger.debug("Outlier filter: dropped %d of %d samples", dropped, len(all_features))
        return filtered if len(filtered) >= 5 else all_features

    # ─────────────────────────────────────────────────────
    # ENROLLMENT
    # ─────────────────────────────────────────────────────
    def enroll(self, all_features):
        """
        Build the user's typing profile:
          1. Drop outlier attempts (IQR)
          2. Register bigrams in Bloom filter
          3. Compute per-feature mean + MAD (mean absolute deviation)
        """
        # Step 1 — outlier filter
        clean = self._filter_outliers(all_features)

        # Step 2 — establish bigram ordering
        bigram_set = set()
        for attempt in clean:
            for f in attempt:
                bigram_set.add(f['bigram'])
        self.bigram_order = sorted(bigram_set)

        # Step 3 — populate Bloom filter
        for bigram in self.bigram_order:
            self.bloom.add(bigram)

        # Step 4 — build feature matrix (samples × features)
        X = np.array([self._to_vector(a) for a in clean])

        # Step 5 — compute mean and mean absolute deviation per feature
        self.mean_vector = X.mean(axis=0)
        self.mad_vector  = np.mean(np.abs(X - self.mean_vector), axis=0)

        # Avoid division by zero: floor MAD at a small value
        self.mad_vector = np.where(self.mad_vector < 1e-4, 1e-4, self.mad_vector)

        # Sanity check: score training data against itself
        train_scores = [self.score(a) for a in clean]
        avg_train    = np.mean(train_scores)
        max_train    = np.max(train_scores)

        logger.debug("Bloom: %d bigrams registered", len(self.bigram_order))
        logger.debug("Profile built: mean + MAD over %d features", len(self.bigram_order) * 2)
        logger.debug("Self-check: avg train score = %.3f, max = %.3f (threshold = %s)",
                     avg_train, max_train, self.threshold)

        return len(clean), len(all_features)

    # ─────────────────────────────────────────────────────
    # BLOOM CHECK
    # ─────────────────────────────────────────────────────
    def _bloom_check(self, features):
        if not features:
            return False, 0.0
        seen     = sum(1 for f in features if self.bloom.contains(f['bigram']))
        coverage = seen / len(features)
        return coverage >= 0.70, coverage

    # ─────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────
    def score(self, features):
        """
        Manhattan (scaled) anomaly score.
        Each feature's deviation from the user's mean is divided by
        that feature's mean absolute deviation — this compensates for
        the fact that dwell times are consistent (small MAD) while
        flight times vary more (large MAD).

        Lower score = closer to user's profile.
        """
        if self.mean_vector is None:
            raise RuntimeError("No profile enrolled yet.")

        vec = self._to_vector(features)
        scaled_distances = np.abs(vec - self.mean_vector) / self.mad_vector

        # Average across features — normalizes for feature count
        return round(float(np.mean(scaled_distances)), 4)

    # ─────────────────────────────────────────────────────
    # AUTHENTICATION
    # ─────────────────────────────────────────────────────
    def authenticate(self, features):
        """
        Three-stage decision:
          1. Bloom filter — quick bigram-membership check
          2. Manhattan (scaled) score
          3. Threshold comparison
        """
        if self.mean_vector is None:
            raise RuntimeError("No profile enrolled yet.")

        # Stage 1 — Bloom filter
        bloom_pass, bloom_cov = self._bloom_check(features)
        if not bloom_pass:
            logger.debug("Bloom REJECTED (%.1f%% coverage)", bloom_cov * 100)
            return False, float('inf'), "bloom_reject"

        # Stage 2 — Manhattan (scaled) scoring
        s        = self.score(features)
        accepted = s <= self.threshold
        reason   = "manhattan_accept" if accepted else "manhattan_reject"

        logger.debug("Bloom OK (%.1f%%) | score=%.4f | threshold=%s | %s",
                     bloom_cov * 100, s, self.threshold,
                     "ACCEPTED" if accepted else "REJECTED")

        return accepted, s, reason


    # ─────────────────────────────────────────────────────
    # LOAD FROM SAVED PROFILE
    # ─────────────────────────────────────────────────────
    def load_from_profile(self, profile):
        """Rehydrate engine state from a saved profile dict."""
        self.bigram_order = profile['bigram_order']
        self.mean_vector  = np.array(profile['mean_vector'])
        self.mad_vector   = np.array(profile['mad_vector'])

        # Rebuild Bloom filter from known bigrams
        for bigram in self.bigram_order:
            self.bloom.add(bigram)

    def update_from_profile(self, profile):
        """Sync engine state after ProfileManager updates the profile."""
        self.mean_vector = np.array(profile['mean_vector'])
        self.mad_vector  = np.array(profile['mad_vector'])

# Backwards compatibility alias
SVMEngine = ManhattanScaledEngine