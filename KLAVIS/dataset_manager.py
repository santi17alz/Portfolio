# dataset_manager.py — CMU-schema data collection & export
# Produces CSV files compatible with Killourhy & Maxion 2009 evaluation scripts
import csv
import json
import os
from pathlib import Path
from datetime import datetime


_HERE = Path(__file__).parent
DATASETS_DIR = str(_HERE / "profiles" / "datasets")
CONSENT_FILE = str(_HERE / "profiles" / "datasets" / "consent_log.json")


class DatasetManager:
    """
    Collects keystroke-timing data from every authentication attempt
    in CMU-compatible CSV format. Supports consent tracking and
    master-dataset export for research use.
    """

    def __init__(self):
        os.makedirs(DATASETS_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────
    # CONSENT TRACKING
    # ─────────────────────────────────────────────────────
    def _load_consent_log(self):
        if not os.path.exists(CONSENT_FILE):
            return {}
        with open(CONSENT_FILE, 'r') as f:
            return json.load(f)

    def has_consented(self, username):
        log = self._load_consent_log()
        return username in log

    def record_consent(self, username, consented=True):
        """Log that a user has signed the consent form."""
        log = self._load_consent_log()
        log[username] = {
            'consented':     bool(consented),
            'timestamp':     datetime.now().isoformat(),
            'schema_version': '1.0',
        }
        with open(CONSENT_FILE, 'w') as f:
            json.dump(log, f, indent=2)

    def revoke_consent(self, username):
        """Mark a user's consent as revoked (also deletes their CSV)."""
        log = self._load_consent_log()
        if username in log:
            log[username]['consented']    = False
            log[username]['revoked_at']   = datetime.now().isoformat()
            with open(CONSENT_FILE, 'w') as f:
                json.dump(log, f, indent=2)
        # Delete raw data
        path = self._user_csv_path(username)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ─────────────────────────────────────────────────────
    # FEATURE EXTRACTION — CMU SCHEMA
    # ─────────────────────────────────────────────────────
    def _extract_cmu_features(self, events):
        """
        Convert raw keyboard events into CMU-schema timing features.

        Returns a dict mapping CMU column names to timing values:
          H.key           — hold time for that key (dwell)
          DD.key1.key2    — keydown-to-keydown latency
          UD.key1.key2    — keyup-to-keydown latency (can be negative)

        All values are in seconds.
        """
        # Get the filtered event list (no backspace/shift/etc)
        from features import IGNORED_KEYS
        clean = [(k, e, t) for k, e, t in events if k not in IGNORED_KEYS]

        # Build lookup tables
        down_times = {}
        up_times   = {}
        for k, e, t in clean:
            if e == 'down':
                down_times[k] = t
            elif e == 'up':
                up_times[k] = t

        # Keydown events in typed order
        down_events = [(k, t) for k, e, t in clean if e == 'down']

        features = {}

        # Hold times (dwell) — H.key
        for k, t_down in down_events:
            if k in up_times:
                features[f"H.{self._safe_key(k)}"] = round(up_times[k] - t_down, 4)

        # Bigram timings — DD and UD
        for i in range(len(down_events) - 1):
            k1, t1_down = down_events[i]
            k2, t2_down = down_events[i + 1]
            if k1 not in up_times:
                continue
            t1_up  = up_times[k1]
            dd_key = f"DD.{self._safe_key(k1)}.{self._safe_key(k2)}"
            ud_key = f"UD.{self._safe_key(k1)}.{self._safe_key(k2)}"
            features[dd_key] = round(t2_down - t1_down, 4)
            features[ud_key] = round(t2_down - t1_up,   4)  # can be negative

        return features

    def _safe_key(self, key):
        """Turn key characters into CMU-safe labels (no spaces/special chars)."""
        mapping = {
            ' ': 'space',
            '.': 'period',
            ',': 'comma',
            ';': 'semicolon',
            "'": 'apostrophe',
            '"': 'quote',
            '-': 'dash',
            '_': 'underscore',
            '!': 'bang',
            '?': 'question',
            '/': 'slash',
            '\\': 'backslash',
        }
        return mapping.get(key, key)

    # ─────────────────────────────────────────────────────
    # CSV WRITING
    # ─────────────────────────────────────────────────────
    def _user_csv_path(self, username):
        safe = "".join(c for c in username if c.isalnum() or c in ('_', '-')).lower()
        return os.path.join(DATASETS_DIR, f"{safe}_raw.csv")

    def record_attempt(self, username, events, session_id, rep,
                       attempt_type, accepted=None, label=None):
        """
        Append one keystroke attempt to this user's raw CSV file.

        Parameters
        ----------
        username      : the real user whose PROFILE we're authenticating against
        events        : raw pynput event list
        session_id    : a stable ID for this session (date-based)
        rep           : repetition number within the session
        attempt_type  : 'enroll' | 'auth' | 'bench_genuine' | 'bench_impostor'
        accepted      : auth decision, if applicable
        label         : impostor label, if applicable
        """
        # Gate on consent
        if not self.has_consented(username):
            return False

        features = self._extract_cmu_features(events)
        if not features:
            return False

        path = self._user_csv_path(username)
        write_header = not os.path.exists(path)

        # Fixed metadata columns + all feature columns in sorted order
        meta_cols = ['subject', 'timestamp', 'sessionIndex', 'rep',
                     'attempt_type', 'accepted', 'label']
        feature_cols = sorted(features.keys())

        # Existing file — union the header with any previously unseen columns
        existing_cols = []
        if not write_header:
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_cols = next(reader, [])
            # Combine: keep existing order, append new columns at end
            new_cols = [c for c in feature_cols if c not in existing_cols]
            all_cols = existing_cols + new_cols
            # If there are new cols, we need to rewrite the file with updated header
            if new_cols:
                self._add_columns_to_csv(path, existing_cols, all_cols)
                feature_cols = [c for c in all_cols if c not in meta_cols]
        else:
            all_cols = meta_cols + feature_cols

        row = {
            'subject':      username,
            'timestamp':    datetime.now().isoformat(),
            'sessionIndex': session_id,
            'rep':          rep,
            'attempt_type': attempt_type,
            'accepted':     '' if accepted is None else int(bool(accepted)),
            'label':        label or '',
        }
        row.update(features)

        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            if write_header:
                writer.writeheader()
            # Fill missing keys with empty string
            full_row = {col: row.get(col, '') for col in all_cols}
            writer.writerow(full_row)

        return True

    def _add_columns_to_csv(self, path, old_cols, new_cols):
        """Rewrite an existing CSV with new columns added (filled with empty)."""
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, '') for c in new_cols})

    # ─────────────────────────────────────────────────────
    # MASTER EXPORT
    # ─────────────────────────────────────────────────────
    def export_master_dataset(self, output_path=None):
        """
        Merge every user's raw CSV into one master dataset file.
        Uses the union of all feature columns across users.
        """
        if output_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(DATASETS_DIR,
                                       f"master_dataset_{stamp}.csv")

        # Collect all CSV files
        csv_files = [os.path.join(DATASETS_DIR, f)
                     for f in os.listdir(DATASETS_DIR)
                     if f.endswith('_raw.csv')]

        if not csv_files:
            return None, 0

        # Figure out the union of all columns
        all_cols_set = set()
        for csv_file in csv_files:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, [])
                all_cols_set.update(header)

        # Order: meta columns first, then feature columns sorted
        meta_order = ['subject', 'timestamp', 'sessionIndex', 'rep',
                      'attempt_type', 'accepted', 'label']
        meta_cols    = [c for c in meta_order if c in all_cols_set]
        feature_cols = sorted(c for c in all_cols_set if c not in meta_order)
        all_cols     = meta_cols + feature_cols

        # Write merged
        total_rows = 0
        with open(output_path, 'w', newline='') as outf:
            writer = csv.DictWriter(outf, fieldnames=all_cols)
            writer.writeheader()
            for csv_file in csv_files:
                with open(csv_file, 'r', newline='') as inf:
                    reader = csv.DictReader(inf)
                    for row in reader:
                        writer.writerow({c: row.get(c, '') for c in all_cols})
                        total_rows += 1

        return output_path, total_rows

    # ─────────────────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────────────────
    def stats(self):
        """Return a summary of what's in the dataset right now."""
        summary = {
            'users':        [],
            'total_rows':   0,
            'consent_log':  self._load_consent_log(),
        }
        for f in os.listdir(DATASETS_DIR):
            if not f.endswith('_raw.csv'):
                continue
            path = os.path.join(DATASETS_DIR, f)
            username = f.replace('_raw.csv', '')
            with open(path, 'r', newline='') as csvf:
                reader = csv.reader(csvf)
                next(reader, None)  # skip header
                row_count = sum(1 for _ in reader)
            summary['users'].append({
                'username':  username,
                'row_count': row_count,
                'file_path': path,
            })
            summary['total_rows'] += row_count
        return summary


# Session ID helper — a date-stable string
def make_session_id():
    """Return a session ID that changes at midnight (matches CMU paper's approach)."""
    return datetime.now().strftime("%Y%m%d")