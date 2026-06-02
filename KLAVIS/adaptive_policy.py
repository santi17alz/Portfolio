# adaptive_policy.py — Enrollment stability checker
# All tuning constants live in main_gui.py; this module is pure computation.
import numpy as np


def enrollment_is_stable(engine_class, enroll_features,
                          min_samples, max_samples,
                          window, cv_limit, score_limit):
    """
    Decide whether initial enrollment has collected enough samples.

    Returns
    -------
    done : bool
    info : dict  — explanation for GUI display and profile metadata
    """
    n = len(enroll_features)

    if n < min_samples:
        return False, {
            "reason": "below_minimum",
            "samples": n,
            "target": min_samples,
        }

    if n >= max_samples:
        return True, {
            "reason": "hit_maximum",
            "samples": n,
        }

    # Build a temporary engine to measure self-score stability
    temp_engine = engine_class()
    temp_engine.enroll(enroll_features)

    recent  = enroll_features[-window:]
    vectors = [temp_engine._to_vector(f) for f in recent]
    X       = np.array(vectors)

    means      = np.mean(X, axis=0)
    stds       = np.std(X, axis=0)
    # Only include features with a meaningful mean in the CV calculation.
    # The README uses 1e-4 as the zero-guard, but 1e-3 avoids near-zero
    # bigrams that appear in only 1-2 enrollment samples from inflating CV.
    # The 3-feature minimum prevents false-stable decisions on sparse vectors.
    active_mask = np.abs(means) > 1e-3
    if np.sum(active_mask) < 3:
        # Not enough active features to judge stability yet
        return False, {"reason": "not_stable", "samples": n,
                    "median_cv": 999.0, "avg_recent_score": 999.0}
    active_means = means[active_mask]
    active_stds  = stds[active_mask]
    cvs          = active_stds / np.abs(active_means)
    median_cv    = float(np.median(cvs))

    scores          = [temp_engine.score(f) for f in recent]
    avg_recent_score = float(np.mean(scores))

    stable = median_cv <= cv_limit and avg_recent_score <= score_limit

    return stable, {
        "reason":           "stable" if stable else "not_stable",
        "samples":          n,
        "median_cv":        median_cv,
        "avg_recent_score": avg_recent_score,
    }