# main_gui.py — KLAVIS  |  User-Friendly Redesign
# Changes from original:
#   - Landing screen: Login vs Create Profile choice
#   - Password creation with requirements matching .tie5Roanl complexity
#   - Passphrase IS the user-created password (stored & used for scoring)
#   - Enrollment screen cleaned up — no dev log visible to end user
#   - Certificate integration: checks solo_key_certificate.json,
#     flags retraining if >90 days since first_seen
#   - Dev log hidden behind Ctrl+Shift+D as before
#   - Feature: On-demand re-enrollment (injury / new keyboard) via reset password
#   - Feature: Quarterly re-verification banner + lightweight re-enrollment session
import contextlib
import io
import tkinter as tk
from tkinter import messagebox
import sys
import os
import re
import json
import time
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
import copy
import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    from capture import KeystrokeCapture
    from features import extract_features
    from engine import SVMEngine
    from benchmark import BenchmarkRecorder
    from visualize import BenchmarkVisualizer
    from profile_manager import ProfileManager
    from aggregator import BenchmarkAggregator
    from aggregate_viz import AggregateVisualizer
    from dataset_manager import DatasetManager, make_session_id

try:
    from adaptive_policy import (enrollment_is_stable, MIN_ENROLL, MAX_ENROLL,
                                  STABILITY_WINDOW, STABILITY_CV_LIMIT,
                                  STABILITY_SCORE_LIMIT)
except ImportError:
    # Fallback if adaptive_policy not present
    MIN_ENROLL = 8
    MAX_ENROLL = 20
    STABILITY_WINDOW = 5
    STABILITY_CV_LIMIT = 0.10
    STABILITY_SCORE_LIMIT = 0.80
    def enrollment_is_stable(engine_class, feats, **kwargs):
        n = len(feats)
        if n < MIN_ENROLL:
            return False, {"reason": "below_minimum", "samples": n}
        if n >= MAX_ENROLL:
            return True, {"reason": "hit_maximum", "samples": n}
        return False, {"reason": "not_stable", "samples": n}

# ── Auth thresholds ───────────────────────────────────────
THRESHOLD_TIGHT   = 3.0
THRESHOLD_LOOSE   = 2.5
LEARN_SCORE_LIMIT = 1.5

# ── Certificate ───────────────────────────────────────────
CERT_FILE         = os.path.join("certificates", "solo_key_certificate.json")
RETRAIN_DAYS      = 90   # flag retraining after this many days

# ── Password requirements (mirrors .tie5Roanl complexity) ─
PW_MIN_LEN        = 8
PW_REQUIRE_UPPER  = True
PW_REQUIRE_DIGIT  = True
PW_REQUIRE_SPECIAL = True   # at least one of  . - _ ! @ # $ %

# ── Reset password minimum length (simpler requirement than passphrase) ─
RESET_PW_MIN_LEN  = 8

# ── Quarterly re-enrollment thresholds ───────────────────
QRE_MIN_SAMPLES   = 10
QRE_MAX_SAMPLES   = 15
QRE_DRIFT_LIMIT   = 2.0   # scaled Manhattan distance; above = warn user

# ── Colors ───────────────────────────────────────────────
BG          = "#0a0d14"
PANEL       = "#13161f"
PANEL2      = "#1a1e2b"
BORDER      = "#252838"
ACCENT      = "#4f8ef7"
ACCENT2     = "#a78bfa"
GREEN       = "#34d399"
RED         = "#f87171"
YELLOW      = "#fbbf24"
TEXT        = "#e2e8f0"
TEXT_DIM    = "#4e5670"
TEXT_MED    = "#8892a4"
TEXT_BRIGHT = "#ffffff"
TEAL        = "#2dd4bf"


# ══════════════════════════════════════════════════════════
# CERTIFICATE HELPER  (teammates' implementation — kept intact)
# ══════════════════════════════════════════════════════════
def load_certificate():
    """Return the certificate dict or empty dict if not found."""
    if not os.path.exists(CERT_FILE):
        return {}
    try:
        with open(CERT_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def certificate_needs_retraining():
    """
    Return (needs_retrain: bool, days_old: int | None).
    Checks the first_seen timestamp across all entries.
    """
    cert = load_certificate()
    if not cert:
        return False, None
    oldest = None
    for entry in cert.values():
        ts = entry.get("first_seen")
        if ts:
            dt = datetime.fromisoformat(ts)
            if oldest is None or dt < oldest:
                oldest = dt
    if oldest is None:
        return False, None
    age = (datetime.now() - oldest).days
    return age >= RETRAIN_DAYS, age


# ══════════════════════════════════════════════════════════
# PASSWORD VALIDATION
# ══════════════════════════════════════════════════════════
def validate_password(pw: str):
    """
    Returns (valid: bool, list_of_unmet_requirements: list[str]).
    Requirements mirror the complexity of .tie5Roanl:
      - at least 8 characters
      - at least one uppercase letter
      - at least one digit
      - at least one special character (. - _ ! @ # $ %)
    """
    issues = []
    if len(pw) < PW_MIN_LEN:
        issues.append(f"At least {PW_MIN_LEN} characters")
    if PW_REQUIRE_UPPER and not re.search(r"[A-Z]", pw):
        issues.append("At least one uppercase letter")
    if PW_REQUIRE_DIGIT and not re.search(r"[0-9]", pw):
        issues.append("At least one number")
    if PW_REQUIRE_SPECIAL and not re.search(r"[.\-_!@#$%]", pw):
        issues.append("At least one special character  ( . - _ ! @ # $ % )")
    return len(issues) == 0, issues


# ══════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════
class KlavisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KLAVIS")
        self.root.configure(bg=BG)
        self.root.geometry("780x600")
        self.root.resizable(False, False)

        self.capturer    = KeystrokeCapture()
        self.engine      = SVMEngine(threshold=THRESHOLD_TIGHT)
        self.recorder    = BenchmarkRecorder()
        self.pm          = ProfileManager()
        self.dm          = DatasetManager()
        self.agg         = BenchmarkAggregator()
        self.session_id  = make_session_id()
        self.rep_counter = {}

        # Session state
        self.username        = None
        self.passphrase      = None   # user's chosen password = their passphrase
        self.profile         = None
        self.mode            = None
        self.capturing       = False
        self.enroll_count    = 0
        self.enroll_features = []

        # Enrollment tracking
        self._enroll_is_retrain   = False   # True when called from _start_retraining
        self._pending_reset_data  = None    # set after profile reset, merged at re-enrollment done

        # Re-enrollment session flags
        self.reenrollment_due           = False
        self._reenroll_banner_dismissed = False

        # Quarterly re-enrollment capture state
        self.qre_accepted_features  = []
        self.qre_all_accepted_scores = []
        self.qre_rejected_count     = 0
        self.qre_original_mean      = None
        self.qre_original_mad       = None
        self.qre_counter_label      = None
        self.qre_progress_frame     = None

        # Benchmark state
        self.bench_genuine_count     = 0
        self.bench_genuine_features  = []
        self.bench_genuine_scores    = []
        self.bench_impostor_count    = 0
        self.bench_impostor_features = []
        self.bench_impostor_scores   = []
        self.bench_target_genuine    = 5
        self.bench_target_impostor   = 5
        self.bench_frozen_profile    = None

        # Hidden dev log buffer (shown only via Ctrl+Shift+D)
        self._dev_log_buffer = []

        self.root.bind("<Control-Shift-D>", self._open_dev_menu)

        self._build_landing_screen()

    # ══════════════════════════════════════════════════════
    # SHARED UI HELPERS
    # ══════════════════════════════════════════════════════
    def _clear(self):
        for w in self.root.winfo_children():
            w.destroy()

    def _make_header(self, title, subtitle=None, accent=ACCENT):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=32, pady=(28, 0))
        tk.Label(bar, text="KLAVIS", bg=BG, fg=ACCENT,
                 font=("Courier", 13, "bold")).pack(side="left")
        tk.Label(bar, text=f"● {title}", bg=BG, fg=accent,
                 font=("Courier", 10, "bold")).pack(side="right")
        if subtitle:
            tk.Label(self.root, text=subtitle, bg=BG, fg=TEXT_DIM,
                     font=("Courier", 8)).pack(anchor="w", padx=32)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=32, pady=(8, 0))

    def _make_panel(self):
        f = tk.Frame(self.root, bg=PANEL,
                     highlightbackground=BORDER, highlightthickness=1)
        f.pack(fill="both", expand=True, padx=32, pady=20)
        return f

    def _btn(self, parent, text, command, color=ACCENT, fg=TEXT_BRIGHT,
             padx=24, pady=10, font_size=10):
        return tk.Button(parent, text=text, command=command,
                         bg=color, fg=fg,
                         font=("Courier", font_size, "bold"),
                         relief="flat", bd=0,
                         padx=padx, pady=pady,
                         cursor="hand2",
                         activebackground=color,
                         activeforeground=fg)

    def _dev_log(self, msg, tag="dim"):
        """Internal dev log — not shown in the user UI."""
        self._dev_log_buffer.append((msg, tag))
        if len(self._dev_log_buffer) > 200:
            self._dev_log_buffer = self._dev_log_buffer[-200:]
        # If dev panel is open, push to it
        if hasattr(self, "_dev_text") and self._dev_text:
            try:
                self._dev_text.config(state="normal")
                col_map = {"green": GREEN, "red": RED, "yellow": YELLOW,
                           "accent": ACCENT, "accent2": ACCENT2,
                           "dim": TEXT_DIM, "bright": TEXT_BRIGHT}
                self._dev_text.insert("end", msg + "\n",
                                      col_map.get(tag, TEXT_DIM))
                self._dev_text.see("end")
                self._dev_text.config(state="disabled")
            except tk.TclError:
                self._dev_text = None

    # ══════════════════════════════════════════════════════
    # SCREEN: LANDING  (Login vs Create Profile)
    # ══════════════════════════════════════════════════════
    def _build_landing_screen(self):
        self._clear()
        self._dev_text = None

        # Check certificate for retraining flag
        needs_retrain, cert_age = certificate_needs_retraining()

        # ── Logo block ──────────────────────────────────
        logo_frame = tk.Frame(self.root, bg=BG)
        logo_frame.pack(pady=(50, 0))
        tk.Label(logo_frame, text="KLAVIS",
                 bg=BG, fg=ACCENT,
                 font=("Courier", 42, "bold")).pack()
        tk.Label(logo_frame,
                 text="KEYSTROKE  ·  BIOMETRIC  ·  AUTHENTICATION",
                 bg=BG, fg=TEXT_DIM,
                 font=("Courier", 8, "bold")).pack(pady=(2, 0))

        tk.Frame(self.root, bg=BORDER, height=1).pack(
            fill="x", padx=60, pady=24)

        # ── Certificate warning banner ───────────────────
        if needs_retrain:
            warn = tk.Frame(self.root, bg="#1f1510",
                            highlightbackground=YELLOW, highlightthickness=1)
            warn.pack(fill="x", padx=60, pady=(0, 16))
            tk.Label(warn,
                     text=f"⚠  Security token certificate is {cert_age} days old — profile retraining recommended",
                     bg="#1f1510", fg=YELLOW,
                     font=("Courier", 8, "bold"),
                     pady=8).pack()

        # ── Action buttons ───────────────────────────────
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=10)

        login_btn = self._btn(btn_frame, "  LOG IN  ",
                              self._show_login_panel,
                              color=ACCENT, padx=40, pady=14, font_size=11)
        login_btn.pack(side="left", padx=12)

        create_btn = self._btn(btn_frame, "  CREATE PROFILE  ",
                               self._show_create_profile_panel,
                               color=PANEL2, fg=ACCENT2,
                               padx=40, pady=14, font_size=11)
        create_btn.pack(side="left", padx=12)

        # ── Existing users hint ──────────────────────────
        existing = self.pm.list_users()
        if existing:
            tk.Frame(self.root, bg=BORDER, height=1).pack(
                fill="x", padx=80, pady=20)
            tk.Label(self.root, text="SAVED PROFILES",
                     bg=BG, fg=TEXT_DIM,
                     font=("Courier", 7, "bold")).pack()
            row = tk.Frame(self.root, bg=BG)
            row.pack(pady=6)
            for u in existing[:6]:   # cap at 6 to avoid overflow
                tk.Button(row, text=u,
                          bg=PANEL2, fg=TEXT_MED,
                          font=("Courier", 9),
                          relief="flat", bd=0,
                          padx=12, pady=5, cursor="hand2",
                          command=lambda x=u: self._quick_login(x)).pack(
                    side="left", padx=4)

        # Dev hint
        tk.Label(self.root, text="Ctrl+Shift+D  developer menu",
                 bg=BG, fg="#1e2235",
                 font=("Courier", 7)).pack(side="bottom", pady=6)

    # ── Login panel (slides in below) ───────────────────
    def _show_login_panel(self):
        self._clear()
        self._make_header("LOG IN", accent=ACCENT)

        panel = self._make_panel()

        tk.Label(panel, text="Username",
                 bg=PANEL, fg=TEXT_MED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=40, pady=(30, 4))

        uframe = tk.Frame(panel, bg=PANEL2,
                          highlightbackground=ACCENT, highlightthickness=1)
        uframe.pack(fill="x", padx=40)
        self.login_user_var = tk.StringVar()
        ue = tk.Entry(uframe, textvariable=self.login_user_var,
                      bg=PANEL2, fg=TEXT_BRIGHT,
                      insertbackground=ACCENT,
                      font=("Courier", 13),
                      relief="flat", bd=10)
        ue.pack(fill="x")
        ue.focus_set()
        ue.bind("<Return>", lambda e: self._on_login_submit())

        self.login_err = tk.Label(panel, text="",
                                  bg=PANEL, fg=RED,
                                  font=("Courier", 8))
        self.login_err.pack(pady=(6, 0))

        btn_row = tk.Frame(panel, bg=PANEL)
        btn_row.pack(pady=20)
        self._btn(btn_row, "CONTINUE →", self._on_login_submit,
                  padx=30, pady=10).pack(side="left", padx=6)
        self._btn(btn_row, "← BACK", self._build_landing_screen,
                  color=BORDER, fg=TEXT_DIM,
                  padx=20, pady=10).pack(side="left", padx=6)

    def _on_login_submit(self):
        username = self.login_user_var.get().strip()
        if not username:
            self.login_err.config(text="Please enter a username.")
            return
        if not self.pm.exists(username):
            self.login_err.config(
                text=f"No profile found for '{username}'. Create one first.")
            return
        self.username = username
        self.profile  = self.pm.load(username)
        # Recover the stored passphrase
        self.passphrase = self.profile.get("passphrase", None)
        if self.passphrase is None:
            # Legacy profile without stored passphrase — fall back
            self.passphrase = ".tie5Roanl"
        self.engine.load_from_profile(self.profile)
        self.mode = "auth"

        # Set quarterly re-enrollment flag (integrates with teammates' certificate check)
        self.reenrollment_due           = self._check_reenrollment_due()
        self._reenroll_banner_dismissed = False

        # Check if certificate says retraining needed (teammates' implementation — kept intact)
        needs_retrain, cert_age = certificate_needs_retraining()
        if needs_retrain:
            self._prompt_retraining(cert_age)
        else:
            self._build_auth_screen()

    def _quick_login(self, username):
        """One-click login from saved profile button."""
        self.login_user_var = tk.StringVar(value=username)
        self.username = username
        self.profile  = self.pm.load(username)
        self.passphrase = self.profile.get("passphrase", ".tie5Roanl")
        self.engine.load_from_profile(self.profile)
        self.mode = "auth"

        # Set quarterly re-enrollment flag
        self.reenrollment_due           = self._check_reenrollment_due()
        self._reenroll_banner_dismissed = False

        needs_retrain, cert_age = certificate_needs_retraining()
        if needs_retrain:
            self._prompt_retraining(cert_age)
        else:
            self._build_auth_screen()

    # ── Retraining prompt ────────────────────────────────
    def _prompt_retraining(self, cert_age):
        dlg = tk.Toplevel(self.root)
        dlg.title("Profile Retraining")
        dlg.configure(bg=BG)
        dlg.geometry("460x240")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="⚠  RETRAINING RECOMMENDED",
                 bg=BG, fg=YELLOW,
                 font=("Courier", 13, "bold")).pack(pady=(24, 8))
        tk.Label(dlg,
                 text=(f"Your security token certificate is {cert_age} days old.\n"
                       "For best accuracy, we recommend re-enrolling your profile.\n"
                       "You can skip this and retrain later from your profile settings."),
                 bg=BG, fg=TEXT_MED,
                 font=("Courier", 9),
                 justify="center", wraplength=380).pack(pady=8)

        btn_row = tk.Frame(dlg, bg=BG)
        btn_row.pack(pady=16)

        def do_retrain():
            dlg.destroy()
            self._start_retraining()

        def skip():
            dlg.destroy()
            self._build_auth_screen()

        self._btn(btn_row, "RETRAIN NOW", do_retrain,
                  color=YELLOW, fg=BG,
                  padx=20, pady=8).pack(side="left", padx=8)
        self._btn(btn_row, "SKIP FOR NOW", skip,
                  color=BORDER, fg=TEXT_DIM,
                  padx=20, pady=8).pack(side="left", padx=8)

    def _start_retraining(self):
        """Reset enrollment state and re-enroll existing user."""
        self.enroll_count    = 0
        self.enroll_features = []
        self.mode            = "enroll"
        self.engine          = SVMEngine(threshold=THRESHOLD_TIGHT)
        # Full retrain clears the quarterly re-verification need
        self.reenrollment_due = False
        self._build_enroll_screen(is_retrain=True)

    # ══════════════════════════════════════════════════════
    # SCREEN: CREATE PROFILE
    # ══════════════════════════════════════════════════════
    def _show_create_profile_panel(self):
        self._clear()
        self._make_header("CREATE PROFILE", accent=ACCENT2)

        panel = self._make_panel()

        tk.Label(panel, text="Choose a username",
                 bg=PANEL, fg=TEXT_MED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=40, pady=(24, 4))

        uframe = tk.Frame(panel, bg=PANEL2,
                          highlightbackground=ACCENT2, highlightthickness=1)
        uframe.pack(fill="x", padx=40)
        self.new_user_var = tk.StringVar()
        ue = tk.Entry(uframe, textvariable=self.new_user_var,
                      bg=PANEL2, fg=TEXT_BRIGHT,
                      insertbackground=ACCENT2,
                      font=("Courier", 13),
                      relief="flat", bd=10)
        ue.pack(fill="x")
        ue.focus_set()

        # Password
        tk.Label(panel, text="Create a passphrase  (this is what you will type to authenticate)",
                 bg=PANEL, fg=TEXT_MED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=40, pady=(16, 4))

        pwframe = tk.Frame(panel, bg=PANEL2,
                           highlightbackground=ACCENT2, highlightthickness=1)
        pwframe.pack(fill="x", padx=40)
        self.new_pw_var = tk.StringVar()
        pwe = tk.Entry(pwframe, textvariable=self.new_pw_var,
                       bg=PANEL2, fg=TEXT_BRIGHT,
                       insertbackground=ACCENT2,
                       font=("Courier", 13),
                       show="•",
                       relief="flat", bd=10)
        pwe.pack(fill="x")
        self.new_pw_var.trace_add("write", self._update_pw_requirements)

        # Requirements checklist
        req_frame = tk.Frame(panel, bg=PANEL)
        req_frame.pack(fill="x", padx=40, pady=(8, 0))
        self._req_labels = {}
        requirements = [
            ("len",     f"At least {PW_MIN_LEN} characters"),
            ("upper",   "One uppercase letter"),
            ("digit",   "One number"),
            ("special", "One special character  ( . - _ ! @ # $ % )"),
        ]
        for key, text in requirements:
            lbl = tk.Label(req_frame, text=f"  ○  {text}",
                           bg=PANEL, fg=TEXT_DIM,
                           font=("Courier", 8),
                           anchor="w")
            lbl.pack(fill="x")
            self._req_labels[key] = lbl

        self.create_err = tk.Label(panel, text="",
                                   bg=PANEL, fg=RED,
                                   font=("Courier", 8))
        self.create_err.pack(pady=(6, 0))

        btn_row = tk.Frame(panel, bg=PANEL)
        btn_row.pack(pady=14)
        self._btn(btn_row, "CREATE  →", self._on_create_submit,
                  color=ACCENT2, padx=30, pady=10).pack(side="left", padx=6)
        self._btn(btn_row, "← BACK", self._build_landing_screen,
                  color=BORDER, fg=TEXT_DIM,
                  padx=20, pady=10).pack(side="left", padx=6)

    def _update_pw_requirements(self, *_):
        pw = self.new_pw_var.get()
        checks = {
            "len":     len(pw) >= PW_MIN_LEN,
            "upper":   bool(re.search(r"[A-Z]", pw)),
            "digit":   bool(re.search(r"[0-9]", pw)),
            "special": bool(re.search(r"[.\-_!@#$%]", pw)),
        }
        labels_text = {
            "len":     f"At least {PW_MIN_LEN} characters",
            "upper":   "One uppercase letter",
            "digit":   "One number",
            "special": "One special character  ( . - _ ! @ # $ % )",
        }
        for key, met in checks.items():
            lbl = self._req_labels[key]
            if met:
                lbl.config(text=f"  ✓  {labels_text[key]}", fg=GREEN)
            else:
                lbl.config(text=f"  ○  {labels_text[key]}", fg=TEXT_DIM)

    def _on_create_submit(self):
        username = self.new_user_var.get().strip()
        pw       = self.new_pw_var.get()

        if not username:
            self.create_err.config(text="Please choose a username.")
            return
        if self.pm.exists(username):
            self.create_err.config(
                text=f"Profile '{username}' already exists. Log in instead.")
            return

        valid, issues = validate_password(pw)
        if not valid:
            self.create_err.config(
                text="Passphrase requirements not met — see checklist above.")
            return

        self.username   = username
        self.passphrase = pw

        # Consent gate
        if not self.dm.has_consented(username):
            self._show_consent_dialog()
        else:
            self.mode = "enroll"
            self._build_enroll_screen()

    # ══════════════════════════════════════════════════════
    # CONSENT DIALOG (unchanged logic, restyled)
    # ══════════════════════════════════════════════════════
    def _show_consent_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Data Collection Consent")
        dlg.configure(bg=BG)
        dlg.geometry("520x480")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="DATA COLLECTION CONSENT",
                 bg=BG, fg=YELLOW,
                 font=("Courier", 12, "bold")).pack(pady=(24, 8))

        text = (
            "KLAVIS is a research project studying keystroke biometrics.\n\n"
            "If you consent, the following will be stored locally on this\n"
            "machine and NOT uploaded anywhere:\n\n"
            "  • Key-press and key-release timings (in seconds)\n"
            "  • Your chosen username\n"
            "  • Accept / reject decisions and scores\n\n"
            "You can revoke consent at any time via the developer menu,\n"
            "which will delete your raw timing data from disk.\n\n"
            "This is a student research project only."
        )
        tf = tk.Frame(dlg, bg=PANEL,
                      highlightbackground=BORDER, highlightthickness=1)
        tf.pack(fill="x", padx=24, pady=8)
        tk.Label(tf, text=text, bg=PANEL, fg=TEXT_MED,
                 font=("Courier", 8),
                 justify="left", anchor="w",
                 padx=16, pady=12).pack(fill="x")

        def accept():
            self.dm.record_consent(self.username, consented=True)
            dlg.destroy()
            self.mode = "enroll"
            self._build_enroll_screen()

        def decline():
            dlg.destroy()
            self.username   = None
            self.passphrase = None
            self._build_landing_screen()

        btn_row = tk.Frame(dlg, bg=BG)
        btn_row.pack(pady=16)
        self._btn(btn_row, "✓  I CONSENT", accept,
                  color=GREEN, padx=24, pady=10).pack(side="left", padx=8)
        self._btn(btn_row, "✗  DECLINE", decline,
                  color=BORDER, fg=TEXT_DIM,
                  padx=24, pady=10).pack(side="left", padx=8)

    # ══════════════════════════════════════════════════════
    # SCREEN: ENROLLMENT  (user-friendly — no dev log)
    # ══════════════════════════════════════════════════════
    def _build_enroll_screen(self, is_retrain=False):
        self._clear()
        self._enroll_is_retrain = is_retrain

        verb = "RETRAINING" if is_retrain else "SETTING UP YOUR PROFILE"
        self._make_header(verb,
                          subtitle="This only takes a minute",
                          accent=ACCENT2)

        panel = self._make_panel()

        # Friendly instruction
        tk.Label(panel,
                 text="Type your passphrase several times so KLAVIS can\nlearn your unique typing rhythm.",
                 bg=PANEL, fg=TEXT_MED,
                 font=("Courier", 10),
                 justify="center").pack(pady=(24, 16))

        # Passphrase display box
        pw_box = tk.Frame(panel, bg=PANEL2,
                          highlightbackground=ACCENT2, highlightthickness=1)
        pw_box.pack(fill="x", padx=60, pady=(0, 8))
        tk.Label(pw_box, text="YOUR PASSPHRASE",
                 bg=PANEL2, fg=TEXT_DIM,
                 font=("Courier", 7, "bold")).pack(pady=(8, 2))
        tk.Label(pw_box,
                 text=self.passphrase,
                 bg=PANEL2, fg=ACCENT2,
                 font=("Courier", 18, "bold"),
                 pady=6).pack()
        tk.Label(pw_box, text="Type it exactly as shown, then press Enter",
                 bg=PANEL2, fg=TEXT_DIM,
                 font=("Courier", 7),
                 pady=(0)).pack(pady=(2, 8))

        # Progress bar area
        self.enroll_progress_frame = tk.Frame(panel, bg=PANEL)
        self.enroll_progress_frame.pack(fill="x", padx=60, pady=(8, 4))
        self._draw_enroll_progress(0)

        # Status label (friendly)
        self.enroll_status = tk.Label(panel,
                                      text="Ready — just start typing!",
                                      bg=PANEL, fg=GREEN,
                                      font=("Courier", 10, "bold"))
        self.enroll_status.pack(pady=(4, 10))

        # Input box
        input_frame = tk.Frame(panel, bg=PANEL2,
                               highlightbackground=ACCENT, highlightthickness=1)
        input_frame.pack(fill="x", padx=60)
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame,
                                    textvariable=self.input_var,
                                    bg=PANEL2, fg=TEXT_BRIGHT,
                                    insertbackground=ACCENT,
                                    font=("Courier", 14),
                                    relief="flat", bd=10,
                                    show="•")
        self.input_entry.pack(fill="x")
        self.input_entry.bind("<Return>",     self._on_enter)
        self.input_entry.bind("<KeyPress>",   self._on_keypress)
        self.input_entry.bind("<KeyRelease>", self._on_keyrelease)
        self.input_entry.focus_set()

        # Bind status_dot alias so shared code path works
        self.status_dot = self.enroll_status

    def _draw_enroll_progress(self, count):
        """Draw a simple dot-based progress indicator."""
        for w in self.enroll_progress_frame.winfo_children():
            w.destroy()
        dots_frame = tk.Frame(self.enroll_progress_frame, bg=PANEL)
        dots_frame.pack()
        tk.Label(dots_frame, text="Progress:",
                 bg=PANEL, fg=TEXT_DIM,
                 font=("Courier", 8)).pack(side="left", padx=(0, 8))
        for i in range(MAX_ENROLL):
            color = ACCENT2 if i < count else BORDER
            tk.Label(dots_frame, text="●",
                     bg=PANEL, fg=color,
                     font=("Courier", 10)).pack(side="left", padx=1)

    # ══════════════════════════════════════════════════════
    # SCREEN: AUTHENTICATION
    # ══════════════════════════════════════════════════════
    def _build_auth_screen(self):
        self._clear()

        phase_info = self.pm.phase_info(self.profile)
        phase      = phase_info["phase"]
        phase_color = GREEN if phase == "adaptation" else YELLOW

        self._make_header(f"WELCOME BACK, {self.username.upper()}",
                          accent=phase_color)

        panel = self._make_panel()

        # ── Quarterly re-verification banner ─────────────
        if self.reenrollment_due and not self._reenroll_banner_dismissed:
            banner_bg = "#1c1a00"
            banner = tk.Frame(panel, bg=banner_bg,
                              highlightbackground=YELLOW, highlightthickness=1)
            banner.pack(fill="x", padx=12, pady=(8, 0))
            brow = tk.Frame(banner, bg=banner_bg)
            brow.pack(fill="x", padx=8, pady=5)
            self._btn(brow,
                      "📅  Quarterly re-verification due — click to start",
                      self._build_quarterly_reenroll_intro,
                      color=banner_bg, fg=YELLOW,
                      padx=4, pady=2, font_size=8).pack(side="left")
            self._btn(brow, "✕", self._dismiss_reenrollment_banner,
                      color=banner_bg, fg=YELLOW,
                      padx=4, pady=2, font_size=8).pack(side="right")

        # Subtle phase pill
        pill_text = "● Profile Active" if phase == "adaptation" else "● Profile Learning"
        pill = tk.Frame(panel, bg=PANEL2,
                        highlightbackground=phase_color, highlightthickness=1)
        pill.pack(anchor="e", padx=30, pady=(12, 0))
        tk.Label(pill, text=pill_text,
                 bg=PANEL2, fg=phase_color,
                 font=("Courier", 8, "bold"),
                 padx=10, pady=4).pack()

        # Passphrase display
        tk.Label(panel, text="TYPE YOUR PASSPHRASE",
                 bg=PANEL, fg=TEXT_DIM,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=40, pady=(20, 4))

        pw_box = tk.Frame(panel, bg=PANEL2,
                          highlightbackground=ACCENT, highlightthickness=1)
        pw_box.pack(fill="x", padx=40, pady=(0, 4))
        tk.Label(pw_box, text=self.passphrase,
                 bg=PANEL2, fg=ACCENT,
                 font=("Courier", 16, "bold"),
                 pady=10).pack()

        # Status
        self.status_dot = tk.Label(panel,
                                   text="● Ready",
                                   bg=PANEL, fg=GREEN,
                                   font=("Courier", 10, "bold"))
        self.status_dot.pack(pady=(6, 4))

        # Input
        input_frame = tk.Frame(panel, bg=PANEL2,
                               highlightbackground=ACCENT, highlightthickness=1)
        input_frame.pack(fill="x", padx=40)
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame,
                                    textvariable=self.input_var,
                                    bg=PANEL2, fg=TEXT_BRIGHT,
                                    insertbackground=ACCENT,
                                    font=("Courier", 14),
                                    relief="flat", bd=10,
                                    show="•")
        self.input_entry.pack(fill="x")
        self.input_entry.bind("<Return>",     self._on_enter)
        self.input_entry.bind("<KeyPress>",   self._on_keypress)
        self.input_entry.bind("<KeyRelease>", self._on_keyrelease)
        self.input_entry.focus_set()

        # Action buttons (including new Reset Profile button)
        btn_row = tk.Frame(panel, bg=PANEL)
        btn_row.pack(pady=(20, 6))
        self._btn(btn_row, "← LOG OUT", self._logout,
                  color=BORDER, fg=TEXT_DIM,
                  padx=16, pady=6, font_size=9).pack(side="left", padx=4)
        self._btn(btn_row, "⚙ PROFILE SETTINGS", self._show_profile_settings,
                  color=PANEL2, fg=TEXT_MED,
                  padx=16, pady=6, font_size=9).pack(side="left", padx=4)
        self._btn(btn_row, "🔄 RESET PROFILE", self._on_reset_profile_click,
                  color=RED, fg=TEXT_BRIGHT,
                  padx=16, pady=6, font_size=9).pack(side="left", padx=4)

        self._dev_log(f"═══ AUTH: {self.username} — {phase.upper()} ═══", "yellow")
        self._dev_log(f"Samples: {phase_info['sample_count']} • Age: {phase_info['age_days']}d", "dim")

    # ══════════════════════════════════════════════════════
    # QUARTERLY RE-ENROLLMENT: CHECK + BANNER DISMISS
    # ══════════════════════════════════════════════════════
    def _check_reenrollment_due(self):
        """
        Return True if quarterly re-enrollment is overdue.
        Checks profile-level last_full_reenrollment first; falls back to
        the teammates' certificate check so the two systems integrate
        without duplication.
        """
        if not self.profile:
            return False
        last_reenroll = self.profile.get("last_full_reenrollment")
        if last_reenroll:
            age = (datetime.now() - datetime.fromisoformat(last_reenroll)).days
            return age >= RETRAIN_DAYS
        # No re-enrollment on record — delegate to teammates' certificate check
        needs, _ = certificate_needs_retraining()
        return needs

    def _dismiss_reenrollment_banner(self):
        """Dismiss the banner for this session; it returns on next login."""
        self._reenroll_banner_dismissed = True
        self._build_auth_screen()

    # ══════════════════════════════════════════════════════
    # QUARTERLY RE-ENROLLMENT: FLOW
    # ══════════════════════════════════════════════════════
    def _build_quarterly_reenroll_intro(self):
        """Explanation screen before the quarterly re-verification session."""
        # Reset all QRE state at intro (user may have backed out and restarted)
        self.qre_accepted_features   = []
        self.qre_all_accepted_scores = []
        self.qre_rejected_count      = 0
        self.qre_original_mean       = None
        self.qre_original_mad        = None

        self._clear()
        self._make_header("QUARTERLY RE-VERIFICATION", accent=YELLOW)

        panel = self._make_panel()

        tk.Label(panel,
                 text="📅  It's been 3 months since your last verification.",
                 bg=PANEL, fg=YELLOW,
                 font=("Courier", 10, "bold")).pack(pady=(30, 8))

        info_box = tk.Frame(panel, bg=PANEL2,
                            highlightbackground=BORDER, highlightthickness=1)
        info_box.pack(fill="x", padx=50, pady=(0, 16))
        tk.Label(info_box,
                 text=(
                     "Type your passphrase 10–15 times to refresh your profile.\n"
                     "Each attempt works exactly like a normal login — the system\n"
                     "will learn from your typing over the session."
                 ),
                 bg=PANEL2, fg=TEXT_MED,
                 font=("Courier", 9),
                 justify="center",
                 pady=14, padx=16).pack()

        btn_row = tk.Frame(panel, bg=PANEL)
        btn_row.pack(pady=16)
        self._btn(btn_row, "▶  START", self._build_quarterly_reenroll_screen,
                  color=YELLOW, fg=BG,
                  padx=24, pady=10).pack(side="left", padx=8)
        self._btn(btn_row, "LATER", self._build_auth_screen,
                  color=BORDER, fg=TEXT_DIM,
                  padx=24, pady=10).pack(side="left", padx=8)

    def _build_quarterly_reenroll_screen(self):
        """Capture screen for the quarterly re-verification session."""
        self.mode = "quarterly_reenroll"
        # Snapshot the current profile mean/mad so we can measure drift at the end
        self.qre_original_mean = self.profile["mean_vector"].copy()
        self.qre_original_mad  = self.profile["mad_vector"].copy()

        self._clear()
        self._make_header("QUARTERLY RE-VERIFICATION",
                          subtitle=f"Type your passphrase {QRE_MIN_SAMPLES}–{QRE_MAX_SAMPLES} times",
                          accent=YELLOW)

        panel = self._make_panel()

        # Passphrase display
        pw_box = tk.Frame(panel, bg=PANEL2,
                          highlightbackground=YELLOW, highlightthickness=1)
        pw_box.pack(fill="x", padx=50, pady=(16, 4))
        tk.Label(pw_box, text="YOUR PASSPHRASE",
                 bg=PANEL2, fg=TEXT_DIM,
                 font=("Courier", 7, "bold")).pack(pady=(8, 2))
        tk.Label(pw_box, text=self.passphrase,
                 bg=PANEL2, fg=YELLOW,
                 font=("Courier", 16, "bold"),
                 pady=6).pack()
        tk.Label(pw_box, text="Type it exactly as shown, then press Enter",
                 bg=PANEL2, fg=TEXT_DIM,
                 font=("Courier", 7)).pack(pady=(0, 8))

        # Progress dots (for accepted attempts)
        self.qre_progress_frame = tk.Frame(panel, bg=PANEL)
        self.qre_progress_frame.pack(fill="x", padx=50, pady=(6, 2))
        self._draw_qre_progress(0)

        # Counter label
        self.qre_counter_label = tk.Label(panel,
                                          text="Accepted: 0  |  Total attempts: 0",
                                          bg=PANEL, fg=TEXT_DIM,
                                          font=("Courier", 8))
        self.qre_counter_label.pack(pady=(2, 2))

        # Status dot
        self.status_dot = tk.Label(panel,
                                   text="● Ready — start typing!",
                                   bg=PANEL, fg=GREEN,
                                   font=("Courier", 10, "bold"))
        self.status_dot.pack(pady=(4, 6))

        # Input field
        input_frame = tk.Frame(panel, bg=PANEL2,
                               highlightbackground=YELLOW, highlightthickness=1)
        input_frame.pack(fill="x", padx=50)
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame,
                                    textvariable=self.input_var,
                                    bg=PANEL2, fg=TEXT_BRIGHT,
                                    insertbackground=YELLOW,
                                    font=("Courier", 14),
                                    relief="flat", bd=10,
                                    show="•")
        self.input_entry.pack(fill="x")
        self.input_entry.bind("<Return>",     self._on_enter)
        self.input_entry.bind("<KeyPress>",   self._on_keypress)
        self.input_entry.bind("<KeyRelease>", self._on_keyrelease)
        self.input_entry.focus_set()

        self._btn(panel, "⟵ LATER", self._go_back_from_quarterly,
                  color=BORDER, fg=TEXT_DIM,
                  padx=16, pady=6, font_size=9).pack(pady=(14, 4))

    def _draw_qre_progress(self, accepted_count):
        """Progress dots for quarterly re-enrollment (filled = accepted)."""
        if not self.qre_progress_frame:
            return
        for w in self.qre_progress_frame.winfo_children():
            w.destroy()
        row = tk.Frame(self.qre_progress_frame, bg=PANEL)
        row.pack()
        tk.Label(row, text="Accepted:",
                 bg=PANEL, fg=TEXT_DIM,
                 font=("Courier", 8)).pack(side="left", padx=(0, 8))
        for i in range(QRE_MAX_SAMPLES):
            color = YELLOW if i < accepted_count else BORDER
            tk.Label(row, text="●",
                     bg=PANEL, fg=color,
                     font=("Courier", 10)).pack(side="left", padx=1)

    def _go_back_from_quarterly(self):
        """Cancel quarterly re-enrollment and return to auth screen."""
        self.mode = "auth"
        self._build_auth_screen()

    def _handle_quarterly_reenroll_attempt(self, features):
        """Process one typing attempt during the quarterly re-verification session."""
        phase     = self.pm.compute_phase(self.profile)
        threshold = THRESHOLD_LOOSE if phase == "growth" else THRESHOLD_TIGHT
        score     = self.engine.score(features)
        accepted  = score <= threshold

        total_attempts = len(self.qre_accepted_features) + self.qre_rejected_count + 1

        if accepted:
            # Run normal EMA adaptation — same alpha=0.05, no special learning rate
            vec = self.engine._to_vector(features)
            self.profile = self.pm.update_adaptation(self.profile, vec)
            self.engine.update_from_profile(self.profile)
            self.pm.save(self.profile)
            self.qre_accepted_features.append(features)
            self.qre_all_accepted_scores.append(float(score))
            self._dev_log(
                f"  [QRE ✓] score={score:.3f}  accepted={len(self.qre_accepted_features)}", "green")
            self.status_dot.config(text="● Good — keep going!", fg=GREEN)
        else:
            self.qre_rejected_count += 1
            self._dev_log(
                f"  [QRE ✗] score={score:.3f}  rejected={self.qre_rejected_count}", "red")
            self.status_dot.config(text="● Try again — not a close match", fg=YELLOW)

        # Update counter and dots
        if self.qre_counter_label:
            self.qre_counter_label.config(
                text=f"Accepted: {len(self.qre_accepted_features)}  |  Total attempts: {total_attempts}")
        self._draw_qre_progress(len(self.qre_accepted_features))

        # Check stability — only accepted attempts count toward the sample requirement
        n_accepted = len(self.qre_accepted_features)
        if n_accepted >= QRE_MIN_SAMPLES:
            done, info = enrollment_is_stable(
                SVMEngine, self.qre_accepted_features,
                min_samples=QRE_MIN_SAMPLES,
                max_samples=QRE_MAX_SAMPLES,
                window=STABILITY_WINDOW,
                cv_limit=STABILITY_CV_LIMIT,
                score_limit=STABILITY_SCORE_LIMIT,
            )
            self._dev_log(f"  [QRE stability] {info['reason']}", "dim")
            if done:
                self.root.after(800, self._complete_quarterly_reenrollment)

    def _complete_quarterly_reenrollment(self):
        """Drift check, then finalize or warn the user."""
        n_accepted = len(self.qre_accepted_features)

        # Compute drift between this session's mean and the pre-session profile mean
        if n_accepted >= 3 and self.qre_original_mean is not None:
            vectors      = [self.engine._to_vector(f) for f in self.qre_accepted_features]
            session_mean = np.mean(np.array(vectors), axis=0)
            mad_safe     = np.where(self.qre_original_mad < 1e-4, 1e-4, self.qre_original_mad)
            drift        = float(np.mean(np.abs(session_mean - self.qre_original_mean) / mad_safe))
            self._dev_log(f"  [QRE drift] {drift:.3f}", "dim")

            if drift > QRE_DRIFT_LIMIT:
                self._show_qre_drift_warning(drift)
                return

        self._finalize_quarterly_reenrollment()

    def _show_qre_drift_warning(self, drift):
        """Warn the user that their typing looks significantly different."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Typing Change Detected")
        dlg.configure(bg=BG)
        dlg.geometry("480x300")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="⚠  TYPING CHANGE DETECTED",
                 bg=BG, fg=YELLOW,
                 font=("Courier", 12, "bold")).pack(pady=(24, 8))
        tk.Label(dlg,
                 text=(
                     f"Your typing has changed noticeably (drift score: {drift:.2f}).\n\n"
                     "If this is due to an injury, new keyboard, or other major\n"
                     "change, consider using 'Reset Profile' instead."
                 ),
                 bg=BG, fg=TEXT_MED,
                 font=("Courier", 9),
                 justify="center", wraplength=400).pack(pady=8)

        btn_row = tk.Frame(dlg, bg=BG)
        btn_row.pack(pady=20)

        def continue_anyway():
            dlg.destroy()
            self._finalize_quarterly_reenrollment()

        def go_to_reset():
            dlg.destroy()
            self.mode = "auth"
            self._on_reset_profile_click()

        self._btn(btn_row, "CONTINUE ANYWAY", continue_anyway,
                  color=ACCENT, padx=16, pady=8).pack(side="left", padx=8)
        self._btn(btn_row, "GO TO FULL RESET", go_to_reset,
                  color=RED, fg=TEXT_BRIGHT,
                  padx=16, pady=8).pack(side="left", padx=8)

    def _finalize_quarterly_reenrollment(self):
        """Persist results, clear reenrollment_due, and show completion screen."""
        n_accepted  = len(self.qre_accepted_features)
        scores      = self.qre_all_accepted_scores

        starting_avg = float(np.mean(scores[:3]))  if len(scores) >= 3 else None
        ending_avg   = float(np.mean(scores[-5:])) if len(scores) >= 5 else None

        # Use whatever timestamp field we're tracking — tied to teammates' 90-day cycle
        self.profile["last_full_reenrollment"] = datetime.now().isoformat()
        self.profile.setdefault("reenrollment_history", []).append({
            "completed_at":       datetime.now().isoformat(),
            "type":               "scheduled_quarterly",
            "samples_collected":  n_accepted,
            "samples_rejected":   self.qre_rejected_count,
            "starting_score_avg": starting_avg,
            "ending_score_avg":   ending_avg,
        })
        self.pm.save(self.profile)

        self.reenrollment_due = False
        self.mode = "auth"

        self._dev_log(
            f"✅ Quarterly re-enrollment done: {n_accepted} accepted, "
            f"{self.qre_rejected_count} rejected", "green")

        # Completion screen
        self._clear()
        self._make_header("RE-VERIFICATION COMPLETE", accent=GREEN)
        panel = self._make_panel()

        tk.Label(panel,
                 text="✓  Re-verification complete.",
                 bg=PANEL, fg=GREEN,
                 font=("Courier", 14, "bold")).pack(pady=(50, 8))
        tk.Label(panel,
                 text="Your next re-verification is in 3 months.",
                 bg=PANEL, fg=TEXT_MED,
                 font=("Courier", 10)).pack(pady=(0, 6))
        tk.Label(panel,
                 text=f"Session: {n_accepted} accepted, {self.qre_rejected_count} rejected",
                 bg=PANEL, fg=TEXT_DIM,
                 font=("Courier", 8)).pack()

        self.root.after(3000, self._build_auth_screen)

    # ══════════════════════════════════════════════════════
    # ON-DEMAND RESET: RESET PASSWORD SCREEN
    # ══════════════════════════════════════════════════════
    def _build_reset_password_screen(self):
        """
        Shown once after initial enrollment so the user can set a recovery
        password.  NOT shown for retraining or post-reset re-enrollments.
        """
        self._clear()
        self._make_header("SET A RESET PASSWORD", accent=ACCENT2)

        panel = self._make_panel()

        info_box = tk.Frame(panel, bg=PANEL2,
                            highlightbackground=BORDER, highlightthickness=1)
        info_box.pack(fill="x", padx=50, pady=(20, 16))
        tk.Label(info_box,
                 text=(
                     "If your typing changes significantly (injury, new keyboard, etc.),\n"
                     "you'll need this password to reset your profile.\n"
                     "Make it something you'll remember.\n\n"
                     "This is NOT your authentication passphrase."
                 ),
                 bg=PANEL2, fg=TEXT_MED,
                 font=("Courier", 9),
                 justify="center",
                 pady=14, padx=16).pack()

        tk.Label(panel, text=f"Reset password  (min {RESET_PW_MIN_LEN} characters)",
                 bg=PANEL, fg=TEXT_MED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=50, pady=(0, 4))

        pf1 = tk.Frame(panel, bg=PANEL2,
                       highlightbackground=ACCENT2, highlightthickness=1)
        pf1.pack(fill="x", padx=50)
        rp_var = tk.StringVar()
        rp_entry = tk.Entry(pf1, textvariable=rp_var,
                            bg=PANEL2, fg=TEXT_BRIGHT,
                            insertbackground=ACCENT2,
                            font=("Courier", 13),
                            show="•", relief="flat", bd=10)
        rp_entry.pack(fill="x")
        rp_entry.focus_set()

        tk.Label(panel, text="Confirm reset password",
                 bg=PANEL, fg=TEXT_MED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=50, pady=(12, 4))

        pf2 = tk.Frame(panel, bg=PANEL2,
                       highlightbackground=ACCENT2, highlightthickness=1)
        pf2.pack(fill="x", padx=50)
        rp_confirm_var = tk.StringVar()
        rp_confirm_entry = tk.Entry(pf2, textvariable=rp_confirm_var,
                                    bg=PANEL2, fg=TEXT_BRIGHT,
                                    insertbackground=ACCENT2,
                                    font=("Courier", 13),
                                    show="•", relief="flat", bd=10)
        rp_confirm_entry.pack(fill="x")

        err_lbl = tk.Label(panel, text="", bg=PANEL, fg=RED,
                           font=("Courier", 8))
        err_lbl.pack(pady=(6, 0))

        btn_row = tk.Frame(panel, bg=PANEL)
        btn_row.pack(pady=14)
        self._btn(btn_row, "SAVE AND CONTINUE →",
                  lambda: self._on_reset_password_submit(rp_var, rp_confirm_var, err_lbl),
                  color=ACCENT2, padx=20, pady=10).pack(side="left", padx=6)

    def _hash_reset_password(self, salt_bytes, password):
        """Return hex-encoded SHA-256(salt_bytes + password)."""
        return hashlib.sha256(salt_bytes + password.encode("utf-8")).hexdigest()

    def _on_reset_password_submit(self, rp_var, rp_confirm_var, err_lbl):
        pw      = rp_var.get()
        confirm = rp_confirm_var.get()

        if len(pw) < RESET_PW_MIN_LEN:
            err_lbl.config(text=f"Password must be at least {RESET_PW_MIN_LEN} characters.")
            return
        if pw != confirm:
            err_lbl.config(text="Passwords do not match.")
            return

        salt       = secrets.token_bytes(16)
        pw_hash    = self._hash_reset_password(salt, pw)

        self.profile["reset_password_hash"] = pw_hash
        self.profile["reset_password_salt"] = salt.hex()
        self.pm.save(self.profile)

        self._dev_log("🔑 Reset password set for profile", "green")

        self.mode = "auth"
        self._build_auth_screen()

    # ══════════════════════════════════════════════════════
    # ON-DEMAND RESET: RESET FLOW
    # ══════════════════════════════════════════════════════
    def _check_reset_locked(self):
        """
        Return (locked, unlock_at, recent_failure_count).
        Locked when >= 3 failures in the past 24 hours.
        """
        failures = self.profile.get("reset_password_failures", [])
        now      = datetime.now()
        cutoff   = now - timedelta(hours=24)
        recent   = [f for f in failures
                    if datetime.fromisoformat(f) > cutoff]
        if len(recent) >= 3:
            oldest_of_recent = min(datetime.fromisoformat(f) for f in recent)
            unlock_at = oldest_of_recent + timedelta(hours=24)
            return True, unlock_at, len(recent)
        return False, None, len(recent)

    def _on_reset_profile_click(self):
        """Entry point for the on-demand reset flow."""
        if not self.profile:
            return

        if not self.profile.get("reset_password_hash"):
            messagebox.showinfo(
                "No Reset Password",
                "No reset password is set for this profile.\n"
                "Log out and create a new profile to set one during enrollment.")
            return

        locked, unlock_at, _ = self._check_reset_locked()
        if locked:
            unlock_str = unlock_at.strftime("%Y-%m-%d %H:%M")
            messagebox.showwarning(
                "Reset Locked",
                f"Too many failed attempts.\n"
                f"Reset is locked until {unlock_str}.")
            return

        self._show_reset_password_dialog()

    def _show_reset_password_dialog(self):
        """Password prompt dialog for the on-demand reset."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Reset Profile")
        dlg.configure(bg=BG)
        dlg.geometry("460x280")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="🔄  RESET PROFILE",
                 bg=BG, fg=RED,
                 font=("Courier", 13, "bold")).pack(pady=(24, 4))

        _, _, recent_count = self._check_reset_locked()
        remaining = 3 - recent_count

        tk.Label(dlg,
                 text=(
                     "Enter your reset password to wipe this profile\n"
                     "and start a fresh enrollment.\n\n"
                     f"Attempts remaining: {remaining} of 3"
                 ),
                 bg=BG, fg=TEXT_MED,
                 font=("Courier", 9),
                 justify="center").pack(pady=8)

        pf = tk.Frame(dlg, bg=PANEL2,
                      highlightbackground=RED, highlightthickness=1)
        pf.pack(fill="x", padx=40, pady=8)
        pw_var = tk.StringVar()
        pe = tk.Entry(pf, textvariable=pw_var,
                      bg=PANEL2, fg=TEXT_BRIGHT,
                      insertbackground=RED,
                      font=("Courier", 13),
                      show="•", relief="flat", bd=10)
        pe.pack(fill="x")
        pe.focus_set()
        pe.bind("<Return>", lambda e: self._verify_reset_password(pw_var, attempts_lbl, dlg))

        attempts_lbl = tk.Label(dlg, text="", bg=BG, fg=RED,
                                font=("Courier", 8))
        attempts_lbl.pack()

        btn_row = tk.Frame(dlg, bg=BG)
        btn_row.pack(pady=12)
        self._btn(btn_row, "VERIFY",
                  lambda: self._verify_reset_password(pw_var, attempts_lbl, dlg),
                  color=RED, fg=TEXT_BRIGHT,
                  padx=20, pady=8).pack(side="left", padx=8)
        self._btn(btn_row, "CANCEL", dlg.destroy,
                  color=BORDER, fg=TEXT_DIM,
                  padx=20, pady=8).pack(side="left", padx=8)

    def _verify_reset_password(self, pw_var, attempts_lbl, dlg):
        """Hash the entered password and compare in constant time."""
        entered = pw_var.get()
        if not entered:
            attempts_lbl.config(text="Please enter your reset password.")
            return

        stored_hash = self.profile.get("reset_password_hash", "")
        salt_hex    = self.profile.get("reset_password_salt", "")
        try:
            salt_bytes = bytes.fromhex(salt_hex)
        except ValueError:
            attempts_lbl.config(text="Profile error: corrupt salt.")
            return

        computed = self._hash_reset_password(salt_bytes, entered)

        # Constant-time comparison — never use == for secret comparison
        if hmac.compare_digest(computed, stored_hash):
            # Correct password — clear failures and proceed to confirmation
            self.profile["reset_password_failures"] = []
            self.pm.save(self.profile)
            dlg.destroy()
            self._confirm_profile_reset()
        else:
            # Wrong password — record failure timestamp
            now_iso = datetime.now().isoformat()
            self.profile.setdefault("reset_password_failures", []).append(now_iso)
            self.pm.save(self.profile)

            locked, unlock_at, recent_count = self._check_reset_locked()
            if locked:
                dlg.destroy()
                unlock_str = unlock_at.strftime("%Y-%m-%d %H:%M")
                messagebox.showwarning(
                    "Reset Locked",
                    f"Too many failed attempts.\n"
                    f"Reset is locked until {unlock_str}.")
            else:
                remaining = 3 - recent_count
                attempts_lbl.config(
                    text=f"Incorrect password.  {remaining} attempt{'s' if remaining != 1 else ''} remaining.")
            pw_var.set("")

    def _confirm_profile_reset(self):
        """Final confirmation dialog — user must type RESET to proceed."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Confirm Reset")
        dlg.configure(bg=BG)
        dlg.geometry("460x260")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="⚠  CONFIRM PROFILE RESET",
                 bg=BG, fg=RED,
                 font=("Courier", 12, "bold")).pack(pady=(24, 8))
        tk.Label(dlg,
                 text=(
                     "This will archive your current profile and start\n"
                     "a fresh enrollment session.\n\n"
                     "Type  RESET  to confirm."
                 ),
                 bg=BG, fg=TEXT_MED,
                 font=("Courier", 9),
                 justify="center").pack(pady=8)

        cf = tk.Frame(dlg, bg=PANEL2,
                      highlightbackground=RED, highlightthickness=1)
        cf.pack(fill="x", padx=60, pady=8)
        confirm_var = tk.StringVar()
        ce = tk.Entry(cf, textvariable=confirm_var,
                      bg=PANEL2, fg=TEXT_BRIGHT,
                      insertbackground=RED,
                      font=("Courier", 13),
                      relief="flat", bd=10)
        ce.pack(fill="x")
        ce.focus_set()

        err_lbl = tk.Label(dlg, text="", bg=BG, fg=RED,
                           font=("Courier", 8))
        err_lbl.pack()

        def do_reset():
            if confirm_var.get().strip() != "RESET":
                err_lbl.config(text="Type exactly  RESET  to confirm.")
                return
            dlg.destroy()
            self._execute_profile_reset()

        ce.bind("<Return>", lambda e: do_reset())
        btn_row = tk.Frame(dlg, bg=BG)
        btn_row.pack(pady=8)
        self._btn(btn_row, "CONFIRM RESET", do_reset,
                  color=RED, fg=TEXT_BRIGHT,
                  padx=16, pady=8).pack(side="left", padx=8)
        self._btn(btn_row, "CANCEL", dlg.destroy,
                  color=BORDER, fg=TEXT_DIM,
                  padx=16, pady=8).pack(side="left", padx=8)

    def _execute_profile_reset(self):
        """
        Archive current profile data, clear active fields, persist, then
        navigate to enrollment so the user can start fresh.
        The reset password and profile_history are carried forward.
        """
        # Build the archive entry
        archived = {
            "archived_at":  datetime.now().isoformat(),
            "reason":       "on_demand_reset",
            "sample_count": self.profile["sample_count"],
            "mean_vector":  self.profile["mean_vector"].tolist(),
            "mad_vector":   self.profile["mad_vector"].tolist(),
            "phase":        self.profile.get("phase", "unknown"),
        }
        self.profile.setdefault("profile_history", []).append(archived)

        # Store fields to carry forward through the fresh enrollment
        self._pending_reset_data = {
            "reset_password_hash": self.profile.get("reset_password_hash"),
            "reset_password_salt": self.profile.get("reset_password_salt"),
            "profile_history":     self.profile.get("profile_history", []),
        }

        # Clear active training data but keep the file on disk (preserves history)
        self.profile["auth_history"]      = []
        self.profile["benchmark_history"] = []
        self.profile["sample_count"]      = 0
        self.profile["created_at"]        = datetime.now().isoformat()
        self.profile.pop("last_full_reenrollment", None)
        self.pm.save(self.profile)

        self._dev_log(f"🔄 Profile reset executed for {self.username}", "yellow")

        # Reset enrollment state and go to enrollment screen
        self.enroll_count    = 0
        self.enroll_features = []
        self.mode            = "enroll"
        self.engine          = SVMEngine(threshold=THRESHOLD_TIGHT)
        self._build_enroll_screen(is_retrain=False)

    # ══════════════════════════════════════════════════════
    # PROFILE SETTINGS  (user-facing settings panel)
    # ══════════════════════════════════════════════════════
    def _show_profile_settings(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Profile Settings")
        dlg.configure(bg=BG)
        dlg.geometry("420x380")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="PROFILE SETTINGS",
                 bg=BG, fg=ACCENT,
                 font=("Courier", 12, "bold")).pack(pady=(24, 4))
        tk.Label(dlg, text=f"User: {self.username}",
                 bg=BG, fg=TEXT_MED,
                 font=("Courier", 9)).pack()

        tk.Frame(dlg, bg=BORDER, height=1).pack(fill="x", padx=24, pady=16)

        # Change passphrase
        self._btn(dlg, "CHANGE PASSPHRASE", self._change_passphrase_flow,
                  color=ACCENT, padx=20, pady=8).pack(pady=6)

        # Manual retrain
        self._btn(dlg, "RETRAIN PROFILE", self._start_retraining,
                  color=PANEL2, fg=ACCENT2,
                  padx=20, pady=8).pack(pady=6)

        tk.Frame(dlg, bg=BORDER, height=1).pack(fill="x", padx=24, pady=16)

        # Certificate info
        cert = load_certificate()
        if cert:
            for vid, info in cert.items():
                ts = info.get("first_seen", "unknown")
                try:
                    age = (datetime.now() - datetime.fromisoformat(ts)).days
                    age_str = f"{age} days ago"
                except Exception:
                    age_str = ts
                tk.Label(dlg, text=f"Token  {vid}  ·  first seen {age_str}",
                         bg=BG, fg=TEXT_DIM,
                         font=("Courier", 8)).pack()
        else:
            tk.Label(dlg, text="No certificate data found",
                     bg=BG, fg=TEXT_DIM,
                     font=("Courier", 8)).pack()

        tk.Frame(dlg, bg=BORDER, height=1).pack(fill="x", padx=24, pady=16)
        self._btn(dlg, "CLOSE", dlg.destroy,
                  color=BORDER, fg=TEXT_DIM,
                  padx=20, pady=8).pack(pady=4)

    def _change_passphrase_flow(self):
        """Prompt user to enter a new passphrase, then re-enroll."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Change Passphrase")
        dlg.configure(bg=BG)
        dlg.geometry("440x340")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="CHANGE PASSPHRASE",
                 bg=BG, fg=ACCENT2,
                 font=("Courier", 12, "bold")).pack(pady=(24, 4))
        tk.Label(dlg,
                 text="Enter a new passphrase. You will need to\nretrain your profile after changing it.",
                 bg=BG, fg=TEXT_MED,
                 font=("Courier", 9), justify="center").pack(pady=8)

        pf = tk.Frame(dlg, bg=PANEL2,
                      highlightbackground=ACCENT2, highlightthickness=1)
        pf.pack(fill="x", padx=32, pady=8)
        pw_var = tk.StringVar()
        pe = tk.Entry(pf, textvariable=pw_var,
                      bg=PANEL2, fg=TEXT_BRIGHT,
                      insertbackground=ACCENT2,
                      font=("Courier", 13),
                      show="•", relief="flat", bd=10)
        pe.pack(fill="x")
        pe.focus_set()

        # Mini requirements
        req_frame = tk.Frame(dlg, bg=BG)
        req_frame.pack(fill="x", padx=32)
        req_labels = {}
        req_defs = [("len", f"≥{PW_MIN_LEN} chars"), ("upper", "A-Z"),
                    ("digit", "0-9"), ("special", ". - _ ! @ # $ %")]
        row = tk.Frame(req_frame, bg=BG)
        row.pack()
        for key, txt in req_defs:
            lbl = tk.Label(row, text=f"○ {txt}", bg=BG, fg=TEXT_DIM,
                           font=("Courier", 8))
            lbl.pack(side="left", padx=6)
            req_labels[key] = lbl

        err_lbl = tk.Label(dlg, text="", bg=BG, fg=RED,
                           font=("Courier", 8))
        err_lbl.pack()

        def on_pw_change(*_):
            pw = pw_var.get()
            checks = {
                "len":     len(pw) >= PW_MIN_LEN,
                "upper":   bool(re.search(r"[A-Z]", pw)),
                "digit":   bool(re.search(r"[0-9]", pw)),
                "special": bool(re.search(r"[.\-_!@#$%]", pw)),
            }
            texts = {"len": f"≥{PW_MIN_LEN} chars", "upper": "A-Z",
                     "digit": "0-9", "special": ". - _ ! @ # $ %"}
            for k, met in checks.items():
                req_labels[k].config(
                    text=f"{'✓' if met else '○'} {texts[k]}",
                    fg=GREEN if met else TEXT_DIM)

        pw_var.trace_add("write", on_pw_change)

        def confirm():
            pw = pw_var.get()
            valid, _ = validate_password(pw)
            if not valid:
                err_lbl.config(text="Passphrase requirements not met.")
                return
            self.passphrase = pw
            dlg.destroy()
            self._start_retraining()

        btn_row = tk.Frame(dlg, bg=BG)
        btn_row.pack(pady=16)
        self._btn(btn_row, "SAVE & RETRAIN", confirm,
                  color=ACCENT2, padx=20, pady=8).pack(side="left", padx=6)
        self._btn(btn_row, "CANCEL", dlg.destroy,
                  color=BORDER, fg=TEXT_DIM,
                  padx=20, pady=8).pack(side="left", padx=6)

    # ══════════════════════════════════════════════════════
    # CAPTURE LOGIC  (unchanged from original)
    # ══════════════════════════════════════════════════════
    def _on_keypress(self, event):
        if event.keysym == "BackSpace":
            if self.capturer.events:
                if self.capturer.events[-1][1] == "up":
                    self.capturer.events.pop()
                if self.capturer.events and self.capturer.events[-1][1] == "down":
                    self.capturer.events.pop()
            return
        if event.keysym in ("Return", "Shift_L", "Shift_R", "Control_L",
                            "Control_R", "Alt_L", "Alt_R",
                            "Meta_L", "Meta_R", "Tab"):
            return
        if not self.capturing:
            self.capturer.events = []
            self.capturing = True
            self.status_dot.config(text="● Typing...", fg=YELLOW)
        key = event.char if event.char else event.keysym
        self.capturer.events.append((key, "down", time.perf_counter()))

    def _on_keyrelease(self, event):
        if not self.capturing:
            return
        if event.keysym in ("Return", "Shift_L", "Shift_R", "Control_L",
                            "Control_R", "Alt_L", "Alt_R",
                            "Meta_L", "Meta_R", "Tab", "BackSpace"):
            return
        key = event.char if event.char else event.keysym
        self.capturer.events.append((key, "up", time.perf_counter()))

    def _on_enter(self, event):
        if self.capturing:
            self._process()
            return "break"

    # ══════════════════════════════════════════════════════
    # PROCESSING
    # ══════════════════════════════════════════════════════
    def _process(self):
        self.capturing = False
        typed = self.input_var.get().strip()
        self.input_var.set("")

        if typed != self.passphrase:
            self.status_dot.config(text="● Wrong passphrase — try again", fg=RED)
            self._dev_log(f"  ✗ Wrong passphrase attempt", "red")
            self.capturer.events = []
            return

        features = extract_features(self.capturer.events)
        if not features:
            self.status_dot.config(text="● No data captured — try again", fg=RED)
            return

        if self.mode == "enroll":
            self._handle_enroll_sample(features)
        elif self.mode == "auth":
            self._handle_auth_attempt(features)
        elif self.mode == "quarterly_reenroll":
            self._handle_quarterly_reenroll_attempt(features)
        elif self.mode == "bench_genuine":
            self._handle_benchmark_genuine(features)
        elif self.mode == "bench_impostor":
            self._handle_benchmark_impostor(features)

    # ══════════════════════════════════════════════════════
    # ENROLLMENT LOGIC
    # ══════════════════════════════════════════════════════
    def _handle_enroll_sample(self, features):
        self._record_to_dataset("enroll", self.capturer.events, accepted=True)
        self.enroll_features.append(features)
        self.enroll_count += 1

        done, info = enrollment_is_stable(
            SVMEngine, self.enroll_features,
            min_samples=MIN_ENROLL, max_samples=MAX_ENROLL,
            window=STABILITY_WINDOW, cv_limit=STABILITY_CV_LIMIT,
            score_limit=STABILITY_SCORE_LIMIT,
        )

        self._dev_log(f"  ✓ Sample {self.enroll_count} — {info['reason']}", "green")
        self._draw_enroll_progress(self.enroll_count)

        if done:
            # Snapshot reset password / history from the OLD profile before replacing it.
            # For brand-new users self.profile is None so preserved_fields is empty.
            preserved_fields = {}
            if self.profile:
                for key in ("reset_password_hash", "reset_password_salt",
                            "profile_history", "reenrollment_history",
                            "last_full_reenrollment"):
                    if key in self.profile:
                        preserved_fields[key] = self.profile[key]

            enrollment_policy = {
                "type":              "adaptive",
                "min_samples":       MIN_ENROLL,
                "max_samples":       MAX_ENROLL,
                "stop_reason":       info["reason"],
                "stability_metrics": info,
            }
            self.engine.enroll(self.enroll_features)
            self.profile = self.pm.create_from_enrollment(
                self.username, self.engine,
                len(self.enroll_features),
                enrollment_policy=enrollment_policy,
            )
            self.profile["passphrase"] = self.passphrase

            if self._pending_reset_data:
                # Post-reset re-enrollment: restore archived fields
                self.profile.update(self._pending_reset_data)
                self._pending_reset_data = None
                self.pm.save(self.profile)
                self._dev_log(
                    f"✅ Profile rebuilt after reset — {info['reason']} at {self.enroll_count} samples",
                    "green")
                self.enroll_status.config(text="✓ Profile rebuilt! Taking you to login…", fg=GREEN)
                self.mode = "auth"
                self.root.after(1800, self._build_auth_screen)

            elif self._enroll_is_retrain:
                # Retrain: carry forward reset password, history, and re-enrollment tracking
                self.profile.update(preserved_fields)
                self.pm.save(self.profile)
                self._dev_log(
                    f"✅ Profile retrained — {info['reason']} at {self.enroll_count} samples",
                    "green")
                self.enroll_status.config(text="✓ Profile updated! Taking you to login…", fg=GREEN)
                self.mode = "auth"
                self.root.after(1800, self._build_auth_screen)

            else:
                # First-time enrollment: show reset password screen before finalizing
                self._dev_log(
                    f"✅ Enrollment complete — {info['reason']} at {self.enroll_count} samples, "
                    "prompting for reset password", "green")
                self.enroll_status.config(text="✓ Almost done! Set a reset password…", fg=GREEN)
                self.root.after(800, self._build_reset_password_screen)

        else:
            remaining = MAX_ENROLL - self.enroll_count
            self.enroll_status.config(
                text=f"Good — keep going  ({remaining} more at most)",
                fg=GREEN)

    # ══════════════════════════════════════════════════════
    # AUTH LOGIC  (unchanged algorithm, hidden from user)
    # ══════════════════════════════════════════════════════
    def _handle_auth_attempt(self, features):
        phase     = self.pm.compute_phase(self.profile)
        threshold = THRESHOLD_LOOSE if phase == "growth" else THRESHOLD_TIGHT
        score     = self.engine.score(features)
        accepted  = score <= threshold

        symbol = "✅ ACCEPTED" if accepted else "❌ REJECTED"
        tag    = "green" if accepted else "red"
        self._dev_log(f"  {symbol} score={score:.3f} t={threshold}", tag)

        added = False
        if accepted and score <= LEARN_SCORE_LIMIT:
            vec = self.engine._to_vector(features)
            if phase == "growth":
                self.profile = self.pm.update_growth(self.profile, vec)
                self._dev_log("     📈 Added to profile (growth)", "yellow")
            else:
                self.profile = self.pm.update_adaptation(self.profile, vec)
                self._dev_log("     🎯 Drift update (EMA)", "accent")
            self.engine.update_from_profile(self.profile)
            added = True

        self.profile = self.pm.record_auth(self.profile, score, accepted, added)

        # Check / persist phase transition
        new_phase = self.pm.compute_phase(self.profile)
        if new_phase != self.profile.get("phase"):
            reason = self.pm.phase_transition_info(self.profile).get("reason", "")
            self.profile["phase"] = new_phase
            self.profile.setdefault("phase_history", []).append({
                "timestamp": datetime.now().isoformat(),
                "phase":     new_phase,
                "reason":    reason,
            })
            self._dev_log(f"🎉 Phase → {new_phase} ({reason})", "green")
            if new_phase == "adaptation":
                self.root.after(1800, self._build_auth_screen)

        self.pm.save(self.profile)
        self._record_to_dataset("auth", self.capturer.events, accepted=accepted)

        # User-friendly feedback only
        if accepted:
            self.status_dot.config(
                text=f"● Access granted — welcome, {self.username}!", fg=GREEN)
        else:
            self.status_dot.config(
                text="● Not recognized — please try again", fg=RED)

    # ══════════════════════════════════════════════════════
    # BENCHMARK (kept intact, accessed via dev menu)
    # ══════════════════════════════════════════════════════
    def _start_benchmark(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Benchmark")
        dialog.configure(bg=BG)
        dialog.geometry("360x260")
        dialog.transient(self.root)

        tk.Label(dialog, text="BENCHMARK YOUR PROFILE",
                 bg=BG, fg=ACCENT2,
                 font=("Courier", 12, "bold")).pack(pady=20)
        tk.Label(dialog, text="Genuine tests:", bg=BG, fg=TEXT,
                 font=("Courier", 9)).pack(pady=(8, 2))
        gen_var = tk.IntVar(value=5)
        tk.Spinbox(dialog, from_=3, to=20, textvariable=gen_var,
                   width=8, font=("Courier", 10)).pack()
        tk.Label(dialog, text="Impostor tests:", bg=BG, fg=TEXT,
                 font=("Courier", 9)).pack(pady=(12, 2))
        imp_var = tk.IntVar(value=5)
        tk.Spinbox(dialog, from_=3, to=20, textvariable=imp_var,
                   width=8, font=("Courier", 10)).pack()

        def begin():
            self.bench_target_genuine  = gen_var.get()
            self.bench_target_impostor = imp_var.get()
            dialog.destroy()
            self._enter_benchmark()

        self._btn(dialog, "START", begin, padx=24, pady=8).pack(pady=16)

    def _enter_benchmark(self):
        self.bench_frozen_profile    = copy.deepcopy(self.profile)
        self.mode                    = "bench_genuine"
        self.bench_genuine_count     = 0
        self.bench_genuine_features  = []
        self.bench_genuine_scores    = []
        self.bench_impostor_count    = 0
        self.bench_impostor_features = []
        self.bench_impostor_scores   = []
        self.recorder                = BenchmarkRecorder()
        self._build_benchmark_screen()

    def _build_benchmark_screen(self):
        self._clear()
        phase_info = self.pm.phase_info(self.profile)
        if self.mode == "bench_genuine":
            title    = f"BENCHMARK — Genuine ({self.bench_genuine_count}/{self.bench_target_genuine})"
            color    = GREEN
            subtitle = "Type as YOU normally would"
        else:
            title    = f"BENCHMARK — Impostor ({self.bench_impostor_count}/{self.bench_target_impostor})"
            color    = RED
            subtitle = "Have someone else type"

        self._make_header(title, subtitle=subtitle, accent=color)
        panel = self._make_panel()

        snap = tk.Frame(panel, bg=PANEL2,
                        highlightbackground=ACCENT2, highlightthickness=1)
        snap.pack(fill="x", padx=30, pady=(12, 8))
        tk.Label(snap,
                 text=f"🔒 Frozen snapshot: {phase_info['sample_count']} samples · {phase_info['phase']} phase",
                 bg=PANEL2, fg=ACCENT2,
                 font=("Courier", 8, "bold"), pady=6).pack()

        pw_box = tk.Frame(panel, bg=PANEL2,
                          highlightbackground=ACCENT, highlightthickness=1)
        pw_box.pack(fill="x", padx=30, pady=(0, 4))
        tk.Label(pw_box, text=self.passphrase,
                 bg=PANEL2, fg=ACCENT,
                 font=("Courier", 16, "bold"), pady=8).pack()

        self.counter_label = tk.Label(
            panel,
            text=f"Progress: {self.bench_genuine_count}/{self.bench_target_genuine} genuine, "
                 f"{self.bench_impostor_count}/{self.bench_target_impostor} impostor",
            bg=PANEL, fg=TEXT_DIM, font=("Courier", 9))
        self.counter_label.pack(pady=(6, 2))

        self.status_dot = tk.Label(panel, text="● Ready",
                                   bg=PANEL, fg=GREEN,
                                   font=("Courier", 10, "bold"))
        self.status_dot.pack(pady=(0, 4))

        input_frame = tk.Frame(panel, bg=PANEL2,
                               highlightbackground=ACCENT, highlightthickness=1)
        input_frame.pack(fill="x", padx=30)
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame,
                                    textvariable=self.input_var,
                                    bg=PANEL2, fg=TEXT_BRIGHT,
                                    insertbackground=ACCENT,
                                    font=("Courier", 14),
                                    relief="flat", bd=10, show="•")
        self.input_entry.pack(fill="x")
        self.input_entry.bind("<Return>",     self._on_enter)
        self.input_entry.bind("<KeyPress>",   self._on_keypress)
        self.input_entry.bind("<KeyRelease>", self._on_keyrelease)
        self.input_entry.focus_set()

        self._btn(panel, "✕ CANCEL", self._cancel_benchmark,
                  color=BORDER, fg=TEXT_DIM,
                  padx=16, pady=6, font_size=9).pack(pady=(16, 4))

        self._dev_log("═══ BENCHMARK STARTED ═══", "accent2")

    def _cancel_benchmark(self):
        self.mode = "auth"
        self.bench_frozen_profile = None
        self._build_auth_screen()

    def _handle_benchmark_genuine(self, features):
        score     = self.engine.score(features)
        threshold = THRESHOLD_TIGHT if self.pm.compute_phase(self.profile) == "adaptation" else THRESHOLD_LOOSE
        accepted  = score <= threshold
        self.recorder.record_genuine(score)
        self.bench_genuine_features.append(features)
        self.bench_genuine_scores.append(float(score))
        self.bench_genuine_count += 1
        self._record_to_dataset("bench_genuine", self.capturer.events, accepted=accepted)
        tag = "green" if accepted else "red"
        symbol = "✅ ACCEPTED" if accepted else "❌ REJECTED (FRR)"
        self.status_dot.config(text=f"{symbol}  score={score:.3f}", fg=GREEN if accepted else RED)
        self._dev_log(f"  [genuine {self.bench_genuine_count}] {symbol}  score={score:.3f}", tag)
        self.counter_label.config(
            text=f"Progress: {self.bench_genuine_count}/{self.bench_target_genuine} genuine, "
                 f"{self.bench_impostor_count}/{self.bench_target_impostor} impostor")
        if self.bench_genuine_count >= self.bench_target_genuine:
            self.mode = "bench_impostor"
            self._dev_log("→ Now do impostor tests", "yellow")
            self.root.after(1500, self._build_benchmark_screen)

    def _handle_benchmark_impostor(self, features):
        score     = self.engine.score(features)
        threshold = THRESHOLD_TIGHT if self.pm.compute_phase(self.profile) == "adaptation" else THRESHOLD_LOOSE
        accepted  = score <= threshold
        imp_label = f"imp_{self.bench_impostor_count+1}"
        self.recorder.record_impostor(score, label=imp_label)
        self.bench_impostor_features.append(features)
        self.bench_impostor_scores.append(float(score))
        self.bench_impostor_count += 1
        self._record_to_dataset("bench_impostor", self.capturer.events, accepted=accepted, label=imp_label)
        if accepted:
            self.status_dot.config(text=f"⚠ ACCEPTED (FAR)  score={score:.3f}", fg=RED)
            self._dev_log(f"  [impostor {self.bench_impostor_count}] ⚠ SLIPPED IN  score={score:.3f}  [FAR]", "red")
        else:
            self.status_dot.config(text=f"✅ REJECTED  score={score:.3f}", fg=GREEN)
            self._dev_log(f"  [impostor {self.bench_impostor_count}] ✅ REJECTED  score={score:.3f}", "green")
        self.counter_label.config(
            text=f"Progress: {self.bench_genuine_count}/{self.bench_target_genuine} genuine, "
                 f"{self.bench_impostor_count}/{self.bench_target_impostor} impostor")
        if self.bench_impostor_count >= self.bench_target_impostor:
            self.root.after(1500, self._show_benchmark_results)

    def _show_benchmark_results(self):
        phase_info = self.pm.phase_info(self.profile)
        threshold  = THRESHOLD_TIGHT if phase_info["phase"] == "adaptation" else THRESHOLD_LOOSE
        far        = self.recorder.compute_far(threshold)
        frr        = self.recorder.compute_frr(threshold)
        eer, eer_t = self.recorder.compute_eer()

        bench_history = self.profile.setdefault("benchmark_history", [])
        bench_history.append({
            "timestamp":         datetime.now().isoformat(),
            "sample_count":      phase_info["sample_count"],
            "age_days":          phase_info["age_days"],
            "phase":             phase_info["phase"],
            "n_genuine":         self.bench_genuine_count,
            "n_impostor":        self.bench_impostor_count,
            "threshold":         threshold,
            "far":               float(far),
            "frr":               float(frr),
            "eer":               float(eer) if eer is not None else None,
            "eer_threshold":     float(eer_t) if eer_t is not None else None,
            "genuine_scores":    list(self.bench_genuine_scores),
            "impostor_scores":   list(self.bench_impostor_scores),
            "genuine_features":  self.bench_genuine_features,
            "impostor_features": self.bench_impostor_features,
        })
        if len(bench_history) > 50:
            del bench_history[:-50]
        self.pm.save(self.profile)

        self._clear()
        self._make_header("BENCHMARK RESULTS",
                           subtitle=f"{phase_info['sample_count']} samples · {phase_info['phase']}",
                           accent=YELLOW)
        panel = self._make_panel()

        metrics = tk.Frame(panel, bg=PANEL)
        metrics.pack(fill="x", padx=30, pady=(20, 10))
        for label, val, color in [
            ("FAR",       f"{far*100:.1f}%",
             RED if far > 0.2 else GREEN),
            ("FRR",       f"{frr*100:.1f}%",
             RED if frr > 0.2 else GREEN),
            ("EER",       f"{eer*100:.1f}%" if eer else "—",
             YELLOW if eer and eer > 0.10 else GREEN),
            ("THRESHOLD", f"{threshold}", TEXT),
        ]:
            cell = tk.Frame(metrics, bg=PANEL2,
                            highlightbackground=BORDER, highlightthickness=1)
            cell.pack(side="left", expand=True, fill="both", padx=4)
            tk.Label(cell, text=label, bg=PANEL2, fg=TEXT_DIM,
                     font=("Courier", 8, "bold")).pack(pady=(10, 2))
            tk.Label(cell, text=val, bg=PANEL2, fg=color,
                     font=("Courier", 18, "bold")).pack(pady=(0, 10))

        if eer is not None:
            if eer < 0.10:
                interp, col = "✓ Excellent", GREEN
            elif eer < 0.20:
                interp, col = "~ Acceptable — keep using to improve", YELLOW
            else:
                interp, col = "! Needs more samples", RED
            tk.Label(panel, text=interp, bg=PANEL, fg=col,
                     font=("Courier", 11, "bold")).pack(pady=8)

        btn_row = tk.Frame(panel, bg=PANEL)
        btn_row.pack(pady=14)
        self._btn(btn_row, "📊 VIEW CHARTS", self._show_benchmark_charts,
                  color=ACCENT, padx=20, pady=8).pack(side="left", padx=6)
        self._btn(btn_row, "← BACK", self._exit_benchmark,
                  color=BORDER, fg=TEXT_DIM,
                  padx=20, pady=8).pack(side="left", padx=6)

    def _show_benchmark_charts(self):
        charts_dir = os.path.join("profiles", f"{self.username}_charts")
        stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path  = os.path.join(charts_dir,
                                  f"benchmark_{stamp}_s{self.profile.get('sample_count',0)}.png")
        viz = BenchmarkVisualizer(
            recorder=self.recorder,
            engine=self.engine,
            all_enroll_features=[],
            genuine_features=self.bench_genuine_features,
            impostor_features=self.bench_impostor_features,
            threshold=THRESHOLD_TIGHT if self.pm.compute_phase(self.profile) == "adaptation" else THRESHOLD_LOOSE,
            benchmark_history=self.profile.get("benchmark_history", []),
            save_path=save_path,
        )
        viz.show_all()

    def _exit_benchmark(self):
        self.mode = "auth"
        self.bench_frozen_profile = None
        self._build_auth_screen()

    # ══════════════════════════════════════════════════════
    # AGGREGATE STATS  (dev menu only)
    # ══════════════════════════════════════════════════════
    def _show_aggregate_stats(self):
        self.pm.save(self.profile)
        single = self.agg.aggregate_single_user(self.username)
        if single is None or single["n_runs"] == 0:
            messagebox.showinfo("No Data",
                                "Run at least one benchmark first.")
            return
        try:
            cross = self.agg.aggregate_cross_user(self.username, SVMEngine)
        except Exception as e:
            self._dev_log(f"Cross-user pooling failed: {e}", "red")
            cross = None
        charts_dir = os.path.join("profiles", f"{self.username}_charts")
        stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path  = os.path.join(charts_dir, f"aggregate_{stamp}.png")
        viz = AggregateVisualizer(single_data=single, cross_data=cross,
                                  save_path=save_path)
        viz.show_all()

    # ══════════════════════════════════════════════════════
    # DATASET HELPERS
    # ══════════════════════════════════════════════════════
    def _next_rep(self, username):
        key = (username, self.session_id)
        self.rep_counter[key] = self.rep_counter.get(key, 0) + 1
        return self.rep_counter[key]

    def _record_to_dataset(self, attempt_type, events,
                           accepted=None, label=None):
        if not self.username:
            return
        rep = self._next_rep(self.username)
        self.dm.record_attempt(
            username=self.username,
            events=events,
            session_id=self.session_id,
            rep=rep,
            attempt_type=attempt_type,
            accepted=accepted,
            label=label,
        )

    # ══════════════════════════════════════════════════════
    # NAVIGATION
    # ══════════════════════════════════════════════════════
    def _logout(self):
        self.username               = None
        self.passphrase             = None
        self.profile                = None
        self.mode                   = None
        self.enroll_count           = 0
        self.enroll_features        = []
        self.reenrollment_due       = False
        self._reenroll_banner_dismissed = False
        self._pending_reset_data    = None
        self._build_landing_screen()

    # ══════════════════════════════════════════════════════
    # DEV MENU  (Ctrl+Shift+D)
    # ══════════════════════════════════════════════════════
    def _open_dev_menu(self, event=None):
        dev = tk.Toplevel(self.root)
        dev.title("KLAVIS — Dev Menu")
        dev.configure(bg=BG)
        dev.geometry("640x720")
        dev.transient(self.root)

        tk.Label(dev, text="⚠  DEV MENU", bg=BG, fg=YELLOW,
                 font=("Courier", 13, "bold")).pack(pady=16)

        # Live log replay
        log_frame = tk.Frame(dev, bg=PANEL,
                             highlightbackground=BORDER, highlightthickness=1)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        tk.Label(log_frame, text="LIVE LOG", bg=PANEL, fg=TEXT_BRIGHT,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        self._dev_text = tk.Text(log_frame, bg=PANEL, fg=TEXT_DIM,
                                 font=("Courier", 8),
                                 relief="flat", bd=0, wrap="word",
                                 state="disabled", cursor="arrow",
                                 height=12)
        self._dev_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        col_map = {"green": GREEN, "red": RED, "yellow": YELLOW,
                   "accent": ACCENT, "accent2": ACCENT2,
                   "dim": TEXT_DIM, "bright": TEXT_BRIGHT}
        for name, col in col_map.items():
            self._dev_text.tag_config(name, foreground=col)

        # Replay buffered log
        self._dev_text.config(state="normal")
        for msg, tag in self._dev_log_buffer:
            self._dev_text.insert("end", msg + "\n", tag)
        self._dev_text.see("end")
        self._dev_text.config(state="disabled")

        # Profiles list
        tk.Frame(dev, bg=BORDER, height=1).pack(fill="x", padx=16, pady=4)
        tk.Label(dev, text="PROFILES", bg=BG, fg=TEXT_MED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=16)

        for user in self.pm.list_users():
            row = tk.Frame(dev, bg=BG)
            row.pack(fill="x", padx=16, pady=2)
            p    = self.pm.load(user)
            info = self.pm.phase_info(p)
            lbl  = f"{user}  ·  {info['sample_count']} samples  ·  {info['phase']}"
            tk.Label(row, text=lbl, bg=BG, fg=TEXT, font=("Courier", 8)).pack(side="left")
            tk.Button(row, text="DELETE", bg=RED, fg=TEXT_BRIGHT,
                      font=("Courier", 7, "bold"),
                      relief="flat", bd=0, padx=8, pady=2, cursor="hand2",
                      command=lambda u=user, d=dev: self._dev_delete(u, d)
                      ).pack(side="right")

        # ── Main actions ─────────────────────────────────
        tk.Frame(dev, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)
        act = tk.Frame(dev, bg=BG)
        act.pack(pady=4)

        if self.profile:
            self._btn(act, "📊 BENCHMARK", self._start_benchmark,
                      color=ACCENT2, padx=14, pady=6, font_size=8
                      ).pack(side="left", padx=4)
            self._btn(act, "📈 ALL-TIME STATS", self._show_aggregate_stats,
                      color=ACCENT, padx=14, pady=6, font_size=8
                      ).pack(side="left", padx=4)

        self._btn(act, "💥 DELETE ALL", lambda: self._dev_delete_all(dev),
                  color=RED, padx=14, pady=6, font_size=8
                  ).pack(side="left", padx=4)
        self._btn(act, "CLOSE", dev.destroy,
                  color=BORDER, fg=TEXT_DIM,
                  padx=14, pady=6, font_size=8
                  ).pack(side="left", padx=4)

        # ── Re-enrollment debug tools ─────────────────────
        tk.Frame(dev, bg=BORDER, height=1).pack(fill="x", padx=16, pady=4)
        tk.Label(dev, text="RE-ENROLLMENT DEBUG", bg=BG, fg=TEXT_MED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=16)

        act2 = tk.Frame(dev, bg=BG)
        act2.pack(pady=4)

        self._btn(act2, "⏰ FORCE RE-ENROLL DUE",
                  lambda: self._dev_force_reenroll_due(dev),
                  color=YELLOW, fg=BG,
                  padx=12, pady=6, font_size=8
                  ).pack(side="left", padx=4)

        if self.profile:
            self._btn(act2, "🔑 CLEAR RESET PW",
                      lambda: self._dev_clear_reset_password(dev),
                      color=PANEL2, fg=TEXT_MED,
                      padx=12, pady=6, font_size=8
                      ).pack(side="left", padx=4)

            if self.profile.get("profile_history"):
                self._btn(act2, "📜 PROFILE HISTORY",
                          self._dev_show_profile_history,
                          color=PANEL2, fg=TEXT_MED,
                          padx=12, pady=6, font_size=8
                          ).pack(side="left", padx=4)

    # ── Dev menu: helpers ───────────────────────────────
    def _dev_delete(self, username, dev_window):
        if messagebox.askyesno("Confirm", f"Delete profile for {username}?"):
            self.pm.delete(username)
            dev_window.destroy()
            self._open_dev_menu()

    def _dev_delete_all(self, dev_window):
        if messagebox.askyesno("Confirm", "Delete ALL profiles?"):
            count = self.pm.delete_all()
            messagebox.showinfo("Done", f"Deleted {count} profiles.")
            dev_window.destroy()
            if self.mode is not None:
                self._logout()

    def _dev_force_reenroll_due(self, dev_window=None):
        """Set last_full_reenrollment to 100 days ago to trigger the quarterly banner."""
        if dev_window:
            dev_window.destroy()
        if self.profile:
            backdated = (datetime.now() - timedelta(days=100)).isoformat()
            self.profile["last_full_reenrollment"] = backdated
            self.pm.save(self.profile)
            self.reenrollment_due           = True
            self._reenroll_banner_dismissed = False
            self._dev_log("⏰ Dev: forced reenrollment_due = True (100 days ago)", "yellow")
            if self.mode == "auth":
                self._build_auth_screen()
            else:
                messagebox.showinfo("Dev", "Re-enrollment flagged. Banner will show on next auth screen.")
        else:
            messagebox.showinfo("Dev", "No profile loaded — log in first.")

    def _dev_clear_reset_password(self, dev_window=None):
        """Remove the reset password fields from the current profile (for testing)."""
        if dev_window:
            dev_window.destroy()
        if self.profile:
            self.profile.pop("reset_password_hash",     None)
            self.profile.pop("reset_password_salt",     None)
            self.profile.pop("reset_password_failures", None)
            self.pm.save(self.profile)
            self._dev_log("🔑 Dev: reset password cleared from profile", "yellow")
            messagebox.showinfo("Dev", "Reset password cleared.")
        else:
            messagebox.showinfo("Dev", "No profile loaded — log in first.")

    def _dev_show_profile_history(self):
        """Read-only window showing all archived profiles in profile_history."""
        history = self.profile.get("profile_history", []) if self.profile else []
        if not history:
            messagebox.showinfo("Profile History", "No archived profiles found.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Profile History")
        dlg.configure(bg=BG)
        dlg.geometry("540x440")
        dlg.transient(self.root)

        tk.Label(dlg, text="PROFILE HISTORY",
                 bg=BG, fg=ACCENT2,
                 font=("Courier", 12, "bold")).pack(pady=(16, 4))
        tk.Label(dlg, text=f"{len(history)} archived snapshot(s)",
                 bg=BG, fg=TEXT_DIM,
                 font=("Courier", 8)).pack()

        frame = tk.Frame(dlg, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=16, pady=12)

        text = tk.Text(frame, bg=PANEL, fg=TEXT_MED,
                       font=("Courier", 8),
                       relief="flat", bd=8, wrap="word",
                       state="disabled", cursor="arrow")
        scrollbar = tk.Scrollbar(frame, command=text.yview)
        text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)

        text.config(state="normal")
        for i, entry in enumerate(history, 1):
            text.insert("end", f"─── Archive {i} ───────────────────────────\n")
            text.insert("end", f"  Archived at : {entry.get('archived_at', 'unknown')}\n")
            text.insert("end", f"  Reason      : {entry.get('reason', 'unknown')}\n")
            text.insert("end", f"  Phase       : {entry.get('phase', 'unknown')}\n")
            text.insert("end", f"  Samples     : {entry.get('sample_count', '?')}\n\n")
        text.config(state="disabled")

        self._btn(dlg, "CLOSE", dlg.destroy,
                  color=BORDER, fg=TEXT_DIM,
                  padx=20, pady=8).pack(pady=8)


# ══════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.lift()
    root.attributes("-topmost", True)
    root.after(100, lambda: root.attributes("-topmost", False))
    root.focus_force()
    KlavisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
