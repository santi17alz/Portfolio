# profile_manager.py — Multi-user profile persistence
# Handles: saving/loading profiles, lifecycle phases, template adaptation
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime


_HERE = Path(__file__).parent
PROFILES_DIR = str(_HERE / "profiles")

# Lifecycle thresholds
# Growth phase — absolute safety limits
MIN_GROWTH_SAMPLES   = 20
MAX_GROWTH_SAMPLES   = 45
MAX_GROWTH_DAYS      = 30

# Phase switching — evidence window
PHASE_HISTORY_WINDOW   = 10
PHASE_ACCEPT_RATE_MIN  = 0.80
PHASE_AVG_SCORE_LIMIT  = 1.50
PHASE_SCORE_STD_LIMIT  = 0.35

# Keep for backward compat (used in phase_info samples_to_go)
GROWTH_SAMPLE_THRESHOLD = MIN_GROWTH_SAMPLES
GROWTH_DAYS_THRESHOLD   = MAX_GROWTH_DAYS

# Phase 2 (adaptation) EMA learning rate
# 0.05 = new sample contributes 5% to mean, old profile keeps 95%
# Research suggests 0.02-0.10 is the useful range
EMA_ALPHA = 0.05


class ProfileManager:
    """
    Manages per-user keystroke biometric profiles on disk.
    Each profile is a JSON file in the profiles/ directory.
    """

    def __init__(self, profiles_dir=PROFILES_DIR):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────
    # FILE OPERATIONS
    # ─────────────────────────────────────────────────────
    def _profile_path(self, username):
        """Sanitize username into a safe filename."""
        safe = "".join(c for c in username if c.isalnum() or c in ('_', '-')).lower()
        assert safe, f"Username '{username}' sanitized to empty string — rejected upstream?"
        return os.path.join(self.profiles_dir, f"{safe}.json")

    def exists(self, username):
        return os.path.exists(self._profile_path(username))

    def list_users(self):
        """Return list of all enrolled usernames."""
        if not os.path.exists(self.profiles_dir):
            return []
        return [
            f.replace(".json", "")
            for f in os.listdir(self.profiles_dir)
            if f.endswith(".json")
        ]

    def delete(self, username):
        """Remove a profile from disk (used by dev menu)."""
        path = self._profile_path(username)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def delete_all(self):
        """Wipe every profile (dev menu nuclear option)."""
        count = 0
        for user in self.list_users():
            if self.delete(user):
                count += 1
        return count

    # ─────────────────────────────────────────────────────
    # LOAD / SAVE
    # ─────────────────────────────────────────────────────
    def load(self, username):
        """Load a profile from disk. Returns None if not found."""
        path = self._profile_path(username)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert arrays back to numpy for the engine
        data['mean_vector'] = np.array(data['mean_vector'])
        data['mad_vector']  = np.array(data['mad_vector'])
        # Always recompute phase from auth_history so it's never stale.
        data['phase'] = self.compute_phase(data)
        return data

    def save(self, profile_data):
        """Save a profile to disk, converting numpy arrays to lists."""
        path = self._profile_path(profile_data['username'])
        # Convert numpy arrays to JSON-serializable lists
        serializable = dict(profile_data)
        serializable['mean_vector']  = profile_data['mean_vector'].tolist()
        serializable['mad_vector']   = profile_data['mad_vector'].tolist()
        serializable['last_updated'] = datetime.now().isoformat()
        # Drop phase — it's always recomputed from auth_history on load.
        serializable.pop('phase', None)
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    # ─────────────────────────────────────────────────────
    # LIFECYCLE PHASE LOGIC
    # ─────────────────────────────────────────────────────
    def phase_transition_info(self, profile):
        """
        Decide whether the profile should be in growth or adaptation.
        Returns a dict with 'phase' and supporting metadata for UI/debugging.
        """
        created  = datetime.fromisoformat(profile['created_at'])
        age_days = (datetime.now() - created).days
        sample_count = profile['sample_count']
        history      = profile.get('auth_history', [])

        # Already in adaptation — stay there
        if profile.get('phase') == 'adaptation':
            return {
                'phase':        'adaptation',
                'reason':       'already_adaptive',
                'sample_count': sample_count,
                'age_days':     age_days,
            }

        # Hard upper limits — force transition
        if sample_count >= MAX_GROWTH_SAMPLES:
            return {
                'phase':        'adaptation',
                'reason':       'max_growth_samples',
                'sample_count': sample_count,
                'age_days':     age_days,
            }

        if age_days >= MAX_GROWTH_DAYS:
            return {
                'phase':        'adaptation',
                'reason':       'max_growth_days',
                'sample_count': sample_count,
                'age_days':     age_days,
            }

        # Not enough samples yet
        if sample_count < MIN_GROWTH_SAMPLES:
            return {
                'phase':        'growth',
                'reason':       'not_enough_samples',
                'sample_count': sample_count,
                'age_days':     age_days,
            }

        # Need enough recent auth history
        recent = history[-PHASE_HISTORY_WINDOW:]
        if len(recent) < PHASE_HISTORY_WINDOW:
            return {
                'phase':        'growth',
                'reason':       'not_enough_recent_auth',
                'sample_count': sample_count,
                'age_days':     age_days,
            }

        # Score quality checks
        accepted     = [h for h in recent if h.get('accepted')]
        accept_rate  = len(accepted) / len(recent)
        scores       = [h['score'] for h in accepted if h.get('score') is not None]

        if not scores:
            return {
                'phase':        'growth',
                'reason':       'no_recent_accepted_scores',
                'sample_count': sample_count,
                'age_days':     age_days,
            }

        avg_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        mature = (
            accept_rate >= PHASE_ACCEPT_RATE_MIN and
            avg_score   <= PHASE_AVG_SCORE_LIMIT and
            std_score   <= PHASE_SCORE_STD_LIMIT
        )

        return {
            'phase':        'adaptation' if mature else 'growth',
            'reason':       'stable_recent_auth' if mature else 'recent_auth_not_stable',
            'sample_count': sample_count,
            'age_days':     age_days,
            'accept_rate':  accept_rate,
            'avg_score':    avg_score,
            'std_score':    std_score,
        }


    def compute_phase(self, profile):
        return self.phase_transition_info(profile)['phase']

    def phase_info(self, profile):
        info = self.phase_transition_info(profile)
        info.update({
            'created_at':    profile['created_at'],
            'last_updated':  profile.get('last_updated', 'never'),
            'samples_to_go': max(0, MIN_GROWTH_SAMPLES - profile['sample_count']),
            'days_to_go':    max(0, MAX_GROWTH_DAYS - info['age_days']),
        })
        return info

    # ─────────────────────────────────────────────────────
    # INITIAL ENROLLMENT
    # ─────────────────────────────────────────────────────
    def create_from_enrollment(self, username, engine, sample_count,
                           enrollment_policy=None):
        """
        Build a new profile from a freshly enrolled engine.
        Engine must have mean_vector, mad_vector, and bigram_order set.
        """
        profile = {
            'username':          username,
            'created_at':        datetime.now().isoformat(),
            'last_updated':      datetime.now().isoformat(),
            'sample_count':      sample_count,
            'bigram_order':      engine.bigram_order,
            'mean_vector':       engine.mean_vector,
            'mad_vector':        engine.mad_vector,
            'auth_history':      [],
            'enrollment_policy': enrollment_policy or {'type': 'fixed'},
        }
        # Set phase in memory so auth comparisons have a non-None baseline.
        # save() strips this field so it is always recomputed fresh on load.
        profile['phase'] = self.compute_phase(profile)
        return profile

    # ─────────────────────────────────────────────────────
    # PROFILE UPDATES — PHASE 1 (GROWTH)
    # ─────────────────────────────────────────────────────
    def update_growth(self, profile, new_sample_vector):
        """
        Phase 1: equal-weight averaging across all samples so far.

        mean_new = (mean_old * N + new_sample) / (N + 1)
        """
        n    = profile['sample_count']
        mean = profile['mean_vector']

        # Rolling mean update
        new_mean = (mean * n + new_sample_vector) / (n + 1)

        # Deviation measured against the OLD mean before it shifted,
        # so the MAD update is not circular.
        new_deviation = np.abs(new_sample_vector - mean)
        new_mad = (profile['mad_vector'] * n + new_deviation) / (n + 1)
        # Floor to prevent division by zero later
        new_mad = np.where(new_mad < 1e-4, 1e-4, new_mad)

        profile['mean_vector']  = new_mean
        profile['mad_vector']   = new_mad
        profile['sample_count'] = n + 1
        return profile

    # ─────────────────────────────────────────────────────
    # PROFILE UPDATES — PHASE 2 (ADAPTATION)
    # ─────────────────────────────────────────────────────
    def update_adaptation(self, profile, new_sample_vector, alpha=EMA_ALPHA):
        """
        Phase 2: exponential moving average — recent samples weighted more.
        This lets the profile gradually track natural drift in typing style.

        mean_new = (1 - alpha) * mean_old + alpha * new_sample
        """
        old_mean = profile['mean_vector']
        new_mean = (1 - alpha) * old_mean + alpha * new_sample_vector

        # Deviation against old_mean to avoid circular reference.
        new_deviation = np.abs(new_sample_vector - old_mean)
        new_mad = (1 - alpha) * profile['mad_vector'] + alpha * new_deviation
        new_mad = np.where(new_mad < 1e-4, 1e-4, new_mad)

        profile['mean_vector']  = new_mean
        profile['mad_vector']   = new_mad
        profile['sample_count'] = profile['sample_count'] + 1
        return profile

    # ─────────────────────────────────────────────────────
    # AUTH HISTORY
    # ─────────────────────────────────────────────────────
    def record_auth(self, profile, score, accepted, added_to_profile):
        """Append an authentication event to the profile's history."""
        profile.setdefault('auth_history', []).append({
            'timestamp':        datetime.now().isoformat(),
            'score':            float(score),
            'accepted':         bool(accepted),
            'added_to_profile': bool(added_to_profile)
        })
        # Keep history to last 100 entries to avoid unbounded file growth
        profile['auth_history'] = profile['auth_history'][-100:]
        return profile