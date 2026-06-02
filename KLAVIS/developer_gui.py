# main_gui.py — KLAVIS Multi-user Keystroke Auth with Lifecycle Profile Management
import contextlib
import io
import tkinter as tk
from tkinter import messagebox, simpledialog
import sys
import os
from pathlib import Path
import time
from datetime import datetime

_HERE = Path(__file__).parent

# Suppress pynput accessibility warning without leaking the redirect on import errors
with contextlib.redirect_stderr(io.StringIO()):
    from capture import KeystrokeCapture
    from features import extract_features
    from engine import ManhattanScaledEngine
    from benchmark import BenchmarkRecorder
    from visualize import BenchmarkVisualizer
    from profile_manager import ProfileManager
    from aggregator import BenchmarkAggregator
    from aggregate_viz import AggregateVisualizer
    from dataset_manager import DatasetManager, make_session_id
    from adaptive_policy import enrollment_is_stable

# ── Config ───────────────────────────────────────────────
PASSPHRASE           = ".tie5Roanl"   # CMU research standard
# Adaptive enrollment config
MIN_ENROLL            = 8
MAX_ENROLL            = 20
STABILITY_WINDOW      = 5
STABILITY_CV_LIMIT    = 0.20
STABILITY_SCORE_LIMIT = 1.25
THRESHOLD_AUTH       = 3.0            # adaptation phase — permissive, profile is mature
THRESHOLD_GROWTH     = 2.5            # growth phase — stricter, only learn from confident samples
LEARN_SCORE_LIMIT    = 1.5            # only add to profile if score < this

# ── Colors ───────────────────────────────────────────────
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
TEXT_BRIGHT = "#ffffff"


class KlavisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KLAVIS — Keystroke Authentication")
        self.root.configure(bg=BG)
        self.root.geometry("820x680")
        self.root.resizable(False, False)

        self.capturer = KeystrokeCapture()
        self.engine   = ManhattanScaledEngine(threshold=THRESHOLD_AUTH)
        self.recorder = BenchmarkRecorder()
        self.pm       = ProfileManager()
        self.dm       = DatasetManager()
        self.agg      = BenchmarkAggregator()
        self.session_id = make_session_id()
        self.rep_counter = {}  # per-session attempt counter per user

        self.username            = None
        self.profile             = None
        self.mode                = None   # "enroll" | "auth" | "bench_genuine" | "bench_impostor"
        self.capturing           = False
        self.enroll_count        = 0
        self.enroll_features     = []

        # Benchmark mode state
        self.bench_genuine_count    = 0
        self.bench_genuine_features = []
        self.bench_impostor_count   = 0
        self.bench_impostor_features = []
        self.bench_target_genuine   = 5
        self.bench_target_impostor  = 5
        self.bench_frozen_profile   = None  # snapshot at bench start

        # Dev menu hotkey
        self.root.bind("<Control-Shift-D>", self._open_dev_menu)

        self._build_login_screen()

    # ═════════════════════════════════════════════════════
    # SCREEN: LOGIN
    # ═════════════════════════════════════════════════════
    def _build_login_screen(self):
        for w in self.root.winfo_children():
            w.destroy()

        # Header
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=30, pady=(40, 0))
        tk.Label(header, text="KLAVIS", bg=BG, fg=ACCENT,
                 font=("Courier", 32, "bold")).pack()
        tk.Label(header, text="KEYSTROKE AUTHENTICATION",
                 bg=BG, fg=TEXT_DIM,
                 font=("Courier", 10, "bold")).pack(pady=(0, 20))

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=30)

        # Main panel
        panel = tk.Frame(self.root, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        panel.pack(fill="both", expand=True, padx=30, pady=30)

        tk.Label(panel, text="Enter your username",
                 bg=PANEL, fg=TEXT_BRIGHT,
                 font=("Courier", 14, "bold")).pack(pady=(40, 4))
        tk.Label(panel, text="New users will be enrolled. Existing users authenticate.",
                 bg=PANEL, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(pady=(0, 30))

        # Username input
        input_frame = tk.Frame(panel, bg=PANEL,
                               highlightbackground=ACCENT, highlightthickness=2)
        input_frame.pack(padx=100, pady=10, fill="x")

        self.username_var = tk.StringVar()
        entry = tk.Entry(input_frame, textvariable=self.username_var,
                         bg="#0f1117", fg=TEXT_BRIGHT,
                         insertbackground=ACCENT,
                         font=("Courier", 14),
                         relief="flat", bd=12, justify="center")
        entry.pack(fill="x")
        entry.bind("<Return>", lambda e: self._on_username_submit())
        entry.focus_set()

        # Continue button
        tk.Button(panel, text="CONTINUE  →",
                  bg=ACCENT, fg=TEXT_BRIGHT,
                  font=("Courier", 11, "bold"),
                  relief="flat", bd=0,
                  padx=30, pady=12,
                  cursor="hand2",
                  command=self._on_username_submit).pack(pady=(20, 10))

        # Existing users list
        existing = self.pm.list_users()
        if existing:
            tk.Frame(panel, bg=BORDER, height=1).pack(fill="x", padx=60, pady=20)
            tk.Label(panel, text="EXISTING PROFILES",
                     bg=PANEL, fg=TEXT_DIM,
                     font=("Courier", 8, "bold")).pack()
            users_frame = tk.Frame(panel, bg=PANEL)
            users_frame.pack(pady=8)
            for user in existing:
                tk.Button(users_frame, text=user,
                          bg=PANEL, fg=ACCENT2,
                          font=("Courier", 10),
                          relief="flat", bd=0, cursor="hand2",
                          command=lambda u=user: self._select_user(u)).pack(side="left", padx=5)

        # Dev hint
        tk.Label(self.root, text="Ctrl+Shift+D for dev menu",
                 bg=BG, fg=TEXT_DIM,
                 font=("Courier", 7)).pack(side="bottom", pady=5)

    def _select_user(self, username):
        self.username_var.set(username)
        self._on_username_submit()

    def _on_username_submit(self):
        username = self.username_var.get().strip()
        if not username:
            return
        if len(username) > 32:
            messagebox.showwarning("Invalid Username",
                                   "Username must be 32 characters or fewer.")
            return
        safe = "".join(c for c in username if c.isalnum() or c in ('_', '-'))
        if not safe:
            messagebox.showwarning("Invalid Username",
                                   "Username must contain at least one letter or number.")
            return
        self.username = username

        if self.pm.exists(username):
            # Existing user → authenticate (consent was collected at signup)
            self.profile = self.pm.load(username)
            self.engine.load_from_profile(self.profile)
            self.mode = "auth"
            self._build_auth_screen()
        else:
            # New user → consent first, then enroll
            if not self.dm.has_consented(username):
                self._show_consent_dialog()
            else:
                self.mode = "enroll"
                self._build_enroll_screen()

    def _show_consent_dialog(self):
        """One-time consent gate before a user can enroll."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Data Collection Consent")
        dlg.configure(bg=BG)
        dlg.geometry("560x540")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="📋 DATA COLLECTION CONSENT",
                 bg=BG, fg=YELLOW,
                 font=("Courier", 13, "bold")).pack(pady=(20, 10))

        tk.Label(dlg, text=f"Hi, {self.username} —",
                 bg=BG, fg=TEXT,
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=30, pady=(0, 8))

        consent_text = (
            "KLAVIS is a research project studying keystroke biometrics.\n\n"
            "If you consent, the following will be collected and stored locally\n"
            "on this machine (NOT uploaded anywhere):\n\n"
            "  • The passphrase you type (known fixed text)\n"
            "  • Key-press and key-release timings (in seconds)\n"
            "  • Your chosen username\n"
            "  • Accept/reject decisions and scores\n\n"
            "You can revoke consent anytime via the dev menu, which will\n"
            "delete your raw timing CSV from disk.\n\n"
            "Data may be exported into a research-format CSV for analysis,\n"
            "but will remain local unless you explicitly share it.\n\n"
            "This is not medical or legal advice. For a student research\n"
            "project only."
        )
        text_frame = tk.Frame(dlg, bg=PANEL,
                              highlightbackground=BORDER, highlightthickness=1)
        text_frame.pack(fill="x", padx=20, pady=10)
        tk.Label(text_frame, text=consent_text,
                 bg=PANEL, fg=TEXT,
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
            self.username = None
            self._build_login_screen()

        btn_frame = tk.Frame(dlg, bg=BG)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="✓ I CONSENT",
                  bg=GREEN, fg=TEXT_BRIGHT,
                  font=("Courier", 10, "bold"),
                  relief="flat", bd=0, padx=24, pady=10, cursor="hand2",
                  command=accept).pack(side="left", padx=8)
        tk.Button(btn_frame, text="✗ DECLINE",
                  bg=BORDER, fg=TEXT_DIM,
                  font=("Courier", 10, "bold"),
                  relief="flat", bd=0, padx=24, pady=10, cursor="hand2",
                  command=decline).pack(side="left", padx=8)

    # ═════════════════════════════════════════════════════
    # SCREEN: ENROLLMENT (first-time users)
    # ═════════════════════════════════════════════════════
    def _build_enroll_screen(self):
        for w in self.root.winfo_children():
            w.destroy()

        # Header
        self._build_header(f"ENROLLMENT — {self.username}",
                           f"Type the passphrase until your profile stabilizes. Min {MIN_ENROLL}, max {MAX_ENROLL} samples.",
                           ACCENT2)

        # Main panel
        panel = self._build_main_panel()

        # Passphrase box
        self._build_passphrase_box(panel)

        # Counter + status
        self.counter_label = tk.Label(panel,
                                      text=f"Attempt 0 of {MAX_ENROLL} max",
                                      bg=PANEL, fg=TEXT_DIM,
                                      font=("Courier", 10))
        self.counter_label.pack(pady=(10, 4))

        self.status_dot = tk.Label(panel,
                                   text="● READY — just start typing!",
                                   bg=PANEL, fg=GREEN,
                                   font=("Courier", 10, "bold"))
        self.status_dot.pack(pady=(0, 12))

        # Input
        self._build_input(panel)

        # Log panel
        self._build_log_panel()
        self._log(f"═══ NEW USER: {self.username} ═══", "accent")
        self._log(f"Type the passphrase until your profile stabilizes", "dim")

    # ═════════════════════════════════════════════════════
    # SCREEN: AUTHENTICATION (existing users)
    # ═════════════════════════════════════════════════════
    def _build_auth_screen(self):
        for w in self.root.winfo_children():
            w.destroy()

        phase_info = self.pm.phase_info(self.profile)
        phase      = phase_info['phase']

        # Header
        phase_label = "GROWTH PHASE" if phase == "growth" else "ADAPTIVE PHASE"
        phase_color = YELLOW if phase == "growth" else GREEN
        self._build_header(f"AUTHENTICATE — {self.username}",
                           f"{phase_label}  •  {phase_info['sample_count']} samples  •  {phase_info['age_days']} days old",
                           phase_color)

        panel = self._build_main_panel()

        # Phase info banner
        banner = tk.Frame(panel, bg="#0f1117",
                          highlightbackground=phase_color, highlightthickness=1)
        banner.pack(fill="x", padx=30, pady=(20, 10))

        if phase == "growth":
            reason = phase_info.get('reason', '')
            banner_text = (f"📈 GROWTH: Profile strengthening. "
                        f"Reason: {reason}. "
                        f"{phase_info['samples_to_go']} more samples "
                        f"or {phase_info['days_to_go']} more days max.")
        else:
            reason = phase_info.get('reason', '')
            banner_text = (f"🎯 ADAPTIVE: Profile mature. "
                        f"Reason: {reason}. Tracking drift with EMA updates.")

        tk.Label(banner, text=banner_text,
                 bg="#0f1117", fg=phase_color,
                 font=("Courier", 9, "bold"),
                 pady=10, wraplength=600).pack()

        # Passphrase
        self._build_passphrase_box(panel)

        self.status_dot = tk.Label(panel,
                                   text="● READY — type to authenticate!",
                                   bg=PANEL, fg=GREEN,
                                   font=("Courier", 10, "bold"))
        self.status_dot.pack(pady=(10, 4))

        self._build_input(panel)

        # Action buttons row
        btn_row = tk.Frame(panel, bg=PANEL)
        btn_row.pack(pady=(20, 10))
        tk.Button(btn_row, text="← LOGOUT",
                  bg=BORDER, fg=TEXT_DIM,
                  font=("Courier", 9, "bold"),
                  relief="flat", bd=0, padx=20, pady=6, cursor="hand2",
                  command=self._logout).pack(side="left", padx=4)
        tk.Button(btn_row, text="📊 TEST PROFILE",
                  bg=ACCENT2, fg=TEXT_BRIGHT,
                  font=("Courier", 9, "bold"),
                  relief="flat", bd=0, padx=20, pady=6, cursor="hand2",
                  command=self._start_benchmark).pack(side="left", padx=4)
        tk.Button(btn_row, text="📈 ALL-TIME STATS",
                  bg=ACCENT, fg=TEXT_BRIGHT,
                  font=("Courier", 9, "bold"),
                  relief="flat", bd=0, padx=20, pady=6, cursor="hand2",
                  command=self._show_aggregate_stats).pack(side="left", padx=4)

        self._build_log_panel()
        self._log(f"═══ {self.username} — {phase.upper()} PHASE ═══",
                  "yellow" if phase == "growth" else "green")
        self._log(f"Samples: {phase_info['sample_count']} • "
                  f"Age: {phase_info['age_days']}d", "dim")

    # ═════════════════════════════════════════════════════
    # UI BUILDERS (shared components)
    # ═════════════════════════════════════════════════════
    def _build_header(self, title, subtitle, color):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=30, pady=(20, 0))
        tk.Label(header, text="KLAVIS", bg=BG, fg=ACCENT,
                 font=("Courier", 14, "bold")).pack(side="left")
        tk.Label(header, text=f"● {title}", bg=BG, fg=color,
                 font=("Courier", 10, "bold")).pack(side="right")
        tk.Label(self.root, text=subtitle, bg=BG, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(padx=30, anchor="w")
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=30, pady=10)

    def _build_main_panel(self):
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True, padx=30)

        panel = tk.Frame(container, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._panel = panel
        self._container = container
        return panel

    def _build_passphrase_box(self, panel):
        tk.Label(panel, text="TYPE THIS PASSPHRASE:",
                 bg=PANEL, fg=TEXT_DIM,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=30, pady=(20, 4))

        box = tk.Frame(panel, bg="#0f1117",
                       highlightbackground=ACCENT, highlightthickness=1)
        box.pack(fill="x", padx=30, pady=(0, 4))
        tk.Label(box, text=PASSPHRASE,
                 bg="#0f1117", fg=ACCENT,
                 font=("Courier", 16, "bold"),
                 pady=10).pack()

    def _build_input(self, panel):
        tk.Label(panel, text="YOUR INPUT:",
                 bg=PANEL, fg=TEXT_DIM,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=30, pady=(12, 4))

        frame = tk.Frame(panel, bg=PANEL,
                         highlightbackground=ACCENT, highlightthickness=1)
        frame.pack(fill="x", padx=30)

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(frame, textvariable=self.input_var,
                                    bg="#0f1117", fg=TEXT_BRIGHT,
                                    insertbackground=ACCENT,
                                    font=("Courier", 14),
                                    relief="flat", bd=10)
        self.input_entry.pack(fill="x")
        self.input_entry.bind("<Return>",     self._on_enter)
        self.input_entry.bind("<KeyPress>",   self._on_keypress)
        self.input_entry.bind("<KeyRelease>", self._on_keyrelease)
        self.input_entry.focus_set()

    def _build_log_panel(self):
        log = tk.Frame(self._container, bg=PANEL,
                       highlightbackground=BORDER, highlightthickness=1,
                       width=240)
        log.pack(side="right", fill="both", padx=(8, 0))
        log.pack_propagate(False)

        tk.Label(log, text="LIVE LOG",
                 bg=PANEL, fg=TEXT_BRIGHT,
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=12, pady=(12, 0))
        tk.Frame(log, bg=BORDER, height=1).pack(fill="x", padx=12, pady=6)

        self.log_text = tk.Text(log, bg=PANEL, fg=TEXT_DIM,
                                font=("Courier", 8),
                                relief="flat", bd=0, wrap="word",
                                state="disabled", cursor="arrow")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

        for name, col in [("green", GREEN), ("red", RED), ("yellow", YELLOW),
                          ("accent", ACCENT), ("accent2", ACCENT2),
                          ("dim", TEXT_DIM), ("bright", TEXT_BRIGHT)]:
            self.log_text.tag_config(name, foreground=col)

    def _log(self, message, tag="dim"):
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # ─────────────────────────────────────────────────────
    # DATASET RECORDING (CMU-schema CSV)
    # ─────────────────────────────────────────────────────
    def _next_rep(self, username):
        """Get the next repetition number for this user in the current session."""
        key = (username, self.session_id)
        self.rep_counter[key] = self.rep_counter.get(key, 0) + 1
        return self.rep_counter[key]

    def _record_to_dataset(self, attempt_type, events, accepted=None, label=None):
        """Record a single attempt to the user's raw CSV if they have consented."""
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
            label=label
        )

    # ═════════════════════════════════════════════════════
    # CAPTURE LOGIC
    # ═════════════════════════════════════════════════════
    def _on_keypress(self, event):
        if event.keysym == 'BackSpace':
            # Walk from the end to find and remove the last typed character's
            # down event (and its matching up event if already recorded).
            events = self.capturer.events
            # Remove trailing up event for the last character, if present
            if events and events[-1][1] == 'up':
                events.pop()
            # Remove the corresponding down event for the same key
            if events and events[-1][1] == 'down':
                events.pop()
            # If neither was found, clear entirely to avoid corruption
            if not events:
                self.capturer.events = []
            return

        if event.keysym in ('Return', 'Shift_L', 'Shift_R', 'Control_L',
                            'Control_R', 'Alt_L', 'Alt_R', 'Meta_L',
                            'Meta_R', 'Tab'):
            return

        if not self.capturing:
            self.capturer.events = []
            self.capturing = True
            self.status_dot.config(text="● CAPTURING...", fg=YELLOW)

        key = event.char if event.char else event.keysym
        self.capturer.events.append((key, 'down', time.perf_counter()))

    def _on_keyrelease(self, event):
        if not self.capturing:
            return
        if event.keysym in ('Return', 'Shift_L', 'Shift_R', 'Control_L',
                            'Control_R', 'Alt_L', 'Alt_R', 'Meta_L',
                            'Meta_R', 'Tab', 'BackSpace'):
            return
        key = event.char if event.char else event.keysym
        self.capturer.events.append((key, 'up', time.perf_counter()))

    def _on_enter(self, event):
        if self.capturing:
            self._process()
            return "break"

    # ═════════════════════════════════════════════════════
    # PROCESSING
    # ═════════════════════════════════════════════════════
    def _process(self):
        self.capturing = False
        typed = self.input_var.get().strip()
        self.input_var.set("")

        if typed.strip() != PASSPHRASE:
            self.status_dot.config(text="● WRONG PASSPHRASE — try again", fg=RED)
            self._log(f"  ✗ Got: '{typed}'", "red")
            self.capturer.events = []
            return

        features = extract_features(self.capturer.events)
        if not features:
            self.status_dot.config(text="● NO DATA — try again", fg=RED)
            return

        if self.mode == "enroll":
            self._handle_enroll_sample(features)
        elif self.mode == "auth":
            self._handle_auth_attempt(features)
        elif self.mode == "bench_genuine":
            self._handle_benchmark_genuine(features)
        elif self.mode == "bench_impostor":
            self._handle_benchmark_impostor(features)

    def _handle_enroll_sample(self, features):
        """First-time enrollment — collect samples adaptively then build profile."""
        self._record_to_dataset("enroll", self.capturer.events, accepted=True)
        self.enroll_features.append(features)
        self.enroll_count += 1

        done, info = enrollment_is_stable(
            ManhattanScaledEngine,
            self.enroll_features,
            min_samples=MIN_ENROLL,
            max_samples=MAX_ENROLL,
            window=STABILITY_WINDOW,
            cv_limit=STABILITY_CV_LIMIT,
            score_limit=STABILITY_SCORE_LIMIT,
        )

        self._log(f"  ✓ Sample {self.enroll_count} — {info['reason']}", "green")
        self.counter_label.config(
            text=f"Attempt {self.enroll_count} of {MAX_ENROLL} max"
        )

        if done:
            enrollment_policy = {
                'type':              'adaptive',
                'min_samples':       MIN_ENROLL,
                'max_samples':       MAX_ENROLL,
                'stop_reason':       info['reason'],
                'stability_metrics': info,
            }

            self.engine.enroll(self.enroll_features)
            self.profile = self.pm.create_from_enrollment(
                self.username,
                self.engine,
                len(self.enroll_features),
                enrollment_policy=enrollment_policy,
            )
            self.pm.save(self.profile)

            self._log(f"✅ Profile created — {info['reason']} at {self.enroll_count} samples", "green")
            self._log(f"   Entering GROWTH phase", "yellow")

            self.mode = "auth"
            self.root.after(1500, self._build_auth_screen)
        else:
            self.status_dot.config(
                text=f"● Keep typing — {info['reason']} ({self.enroll_count}/{MAX_ENROLL})",
                fg=GREEN
            )

    def _handle_auth_attempt(self, features):
        """Existing user — score, decide, maybe update profile."""
        phase      = self.pm.compute_phase(self.profile)
        threshold  = THRESHOLD_GROWTH if phase == "growth" else THRESHOLD_AUTH

        # Score the attempt
        score = self.engine.score(features)
        accepted = score <= threshold

        # Log decision
        symbol = "✅ ACCEPTED" if accepted else "❌ REJECTED"
        tag    = "green" if accepted else "red"
        self._log(f"  {symbol} score={score:.3f} (t={threshold})", tag)

        # Decide whether to learn from this sample
        added = False
        if accepted and score <= LEARN_SCORE_LIMIT:
            # Update profile via the appropriate phase logic
            vec = self.engine._to_vector(features)
            if phase == "growth":
                self.profile = self.pm.update_growth(self.profile, vec)
                self._log(f"     📈 Added to profile (growth)", "yellow")
            else:
                self.profile = self.pm.update_adaptation(self.profile, vec)
                self._log(f"     🎯 Drift update (EMA)", "accent")

            # Sync engine and save
            self.engine.update_from_profile(self.profile)
            added = True
        elif accepted:
            self._log(f"     ○ Not confident enough to learn", "dim")

        # Record history and save
        self.profile = self.pm.record_auth(self.profile, score, accepted, added)
        self.pm.save(self.profile)

        # Dataset CSV
        self._record_to_dataset("auth", self.capturer.events, accepted=accepted)

        new_phase = self.pm.compute_phase(self.profile)
        if new_phase != self.profile.get('phase'):
            reason = self.pm.phase_transition_info(self.profile).get('reason', '')
            self.profile['phase'] = new_phase
            self.profile.setdefault('phase_history', []).append({
                'timestamp': datetime.now().isoformat(),
                'phase':     new_phase,
                'reason':    reason,
            })
            self.pm.save(self.profile)
            if new_phase == "adaptation":
                self._log(f"🎉 Transitioned to ADAPTIVE phase! ({reason})", "green")
                self.root.after(1500, self._build_auth_screen)

        # Update status
        if accepted:
            self.status_dot.config(text=f"● Welcome, {self.username}!", fg=GREEN)
        else:
            self.status_dot.config(text="● Rejected — try again", fg=RED)

    # ═════════════════════════════════════════════════════
    # ALL-TIME AGGREGATE STATS
    # ═════════════════════════════════════════════════════
    def _show_aggregate_stats(self):
        """Build aggregate views and launch the visualizer."""
        import os
        from datetime import datetime

        # Persist any in-memory profile changes first
        self.pm.save(self.profile)

        # Compute single-user aggregation
        single = self.agg.aggregate_single_user(self.username)

        if single is None or single['n_runs'] == 0:
            messagebox.showinfo(
                "No Data Yet",
                "You need to run at least one benchmark first.\n\n"
                "Click 'TEST PROFILE' to run a benchmark."
            )
            return

        # Compute cross-user aggregation (might be empty if you're the only user)
        try:
            cross = self.agg.aggregate_cross_user(self.username, ManhattanScaledEngine)
        except Exception as e:
            self._log(f"Cross-user pooling failed: {e}", "red")
            cross = None

        # Auto-save the chart
        charts_dir = str(_HERE / "profiles" / f"{self.username}_charts")
        stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path  = os.path.join(charts_dir, f"aggregate_{stamp}.png")

        viz = AggregateVisualizer(
            single_data=single,
            cross_data=cross,
            save_path=save_path,
        )

        # Quick log preview before showing chart
        self._log("═══ AGGREGATE STATS ═══", "accent")
        self._log(f"Runs: {single['n_runs']}, "
                  f"Genuine: {single['n_genuine_total']}, "
                  f"Impostor: {single['n_impostor_total']}", "dim")
        if single['pooled_eer'] is not None:
            self._log(f"Pooled EER: {single['pooled_eer']*100:.2f}%", "yellow")
        if cross is not None and cross['n_impostor_total'] > 0:
            self._log(f"Cross-user pool: {cross['n_impostor_total']} impostors "
                      f"from {len(cross['contributing_users'])} other profiles", "accent2")
            if cross['pooled_eer'] is not None:
                self._log(f"Cross-user EER: {cross['pooled_eer']*100:.2f}%", "yellow")

        viz.show_all()

    # ═════════════════════════════════════════════════════
    # BENCHMARK MODE
    # ═════════════════════════════════════════════════════
    def _start_benchmark(self):
        """Ask user how many samples to collect, then begin benchmark."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Benchmark Settings")
        dialog.configure(bg=BG)
        dialog.geometry("380x280")
        dialog.transient(self.root)

        tk.Label(dialog, text="📊 BENCHMARK YOUR PROFILE",
                 bg=BG, fg=ACCENT2,
                 font=("Courier", 12, "bold")).pack(pady=20)

        tk.Label(dialog, text="How many genuine tests?",
                 bg=BG, fg=TEXT,
                 font=("Courier", 9)).pack(pady=(10, 2))
        gen_var = tk.IntVar(value=5)
        tk.Spinbox(dialog, from_=3, to=20, textvariable=gen_var,
                   width=8, font=("Courier", 10)).pack()

        tk.Label(dialog, text="How many impostor tests?",
                 bg=BG, fg=TEXT,
                 font=("Courier", 9)).pack(pady=(15, 2))
        imp_var = tk.IntVar(value=5)
        tk.Spinbox(dialog, from_=3, to=20, textvariable=imp_var,
                   width=8, font=("Courier", 10)).pack()

        tk.Label(dialog, text="(Profile is NOT modified during tests)",
                 bg=BG, fg=TEXT_DIM,
                 font=("Courier", 8, "italic")).pack(pady=(15, 5))

        def begin():
            self.bench_target_genuine  = gen_var.get()
            self.bench_target_impostor = imp_var.get()
            dialog.destroy()
            self._enter_benchmark()

        tk.Button(dialog, text="START",
                  bg=ACCENT, fg=TEXT_BRIGHT,
                  font=("Courier", 10, "bold"),
                  relief="flat", bd=0, padx=24, pady=8, cursor="hand2",
                  command=begin).pack(pady=10)

    def _enter_benchmark(self):
        """Freeze the current profile state & switch to bench mode."""
        import copy
        self.bench_frozen_profile = copy.deepcopy(self.profile)
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
        for w in self.root.winfo_children():
            w.destroy()

        phase_info = self.pm.phase_info(self.profile)
        if self.mode == "bench_genuine":
            title = f"BENCHMARK — Genuine Tests ({self.bench_genuine_count}/{self.bench_target_genuine})"
            color = GREEN
            subtitle = "Type as YOU normally would"
        else:
            title = f"BENCHMARK — Impostor Tests ({self.bench_impostor_count}/{self.bench_target_impostor})"
            color = RED
            subtitle = "Have someone else type (or mimic a different rhythm)"

        self._build_header(title, subtitle, color)

        panel = self._build_main_panel()

        # Snapshot info banner
        snap = tk.Frame(panel, bg="#0f1117",
                        highlightbackground=ACCENT2, highlightthickness=1)
        snap.pack(fill="x", padx=30, pady=(15, 10))
        tk.Label(snap,
                 text=f"🔒 Frozen snapshot: {phase_info['sample_count']} samples • {phase_info['phase']} phase • {phase_info['age_days']}d old",
                 bg="#0f1117", fg=ACCENT2,
                 font=("Courier", 9, "bold"),
                 pady=8, wraplength=600).pack()

        self._build_passphrase_box(panel)

        self.counter_label = tk.Label(panel,
                                      text=f"Progress: {self.bench_genuine_count}/{self.bench_target_genuine} genuine, {self.bench_impostor_count}/{self.bench_target_impostor} impostor",
                                      bg=PANEL, fg=TEXT_DIM,
                                      font=("Courier", 9))
        self.counter_label.pack(pady=(8, 2))

        self.status_dot = tk.Label(panel,
                                   text="● READY — just type!",
                                   bg=PANEL, fg=GREEN,
                                   font=("Courier", 10, "bold"))
        self.status_dot.pack(pady=(0, 4))

        self._build_input(panel)

        # Cancel button
        tk.Button(panel, text="✕ CANCEL BENCHMARK",
                  bg=BORDER, fg=TEXT_DIM,
                  font=("Courier", 9, "bold"),
                  relief="flat", bd=0, padx=20, pady=6, cursor="hand2",
                  command=self._cancel_benchmark).pack(pady=(20, 10))

        self._build_log_panel()
        self._log(f"═══ BENCHMARK STARTED ═══", "accent2")
        self._log(f"Profile snapshot: {phase_info['sample_count']} samples", "dim")
        self._log(f"Target: {self.bench_target_genuine} genuine + {self.bench_target_impostor} impostor", "dim")

    def _cancel_benchmark(self):
        self.mode = "auth"
        self.bench_frozen_profile = None
        self._build_auth_screen()

    def _handle_benchmark_genuine(self, features):
        score = self.engine.score(features)
        threshold = THRESHOLD_AUTH if self.pm.compute_phase(self.profile) == "adaptation" else THRESHOLD_GROWTH
        accepted = score <= threshold

        self.recorder.record_genuine(score)
        self.bench_genuine_features.append(features)
        self.bench_genuine_scores.append(float(score))
        self.bench_genuine_count += 1
        self._record_to_dataset("bench_genuine", self.capturer.events, accepted=accepted)

        # Big on-screen feedback banner (genuine: accepted = correct)
        if accepted:
            self.status_dot.config(text=f"✅ ACCEPTED  score={score:.3f}  (correct — genuine)", fg=GREEN)
            self._log(f"  [genuine {self.bench_genuine_count}] ✅ ACCEPTED  score={score:.3f}", "green")
        else:
            self.status_dot.config(text=f"❌ REJECTED  score={score:.3f}  (false reject — FRR)", fg=RED)
            self._log(f"  [genuine {self.bench_genuine_count}] ❌ REJECTED  score={score:.3f}  [FRR]", "red")

        self.counter_label.config(
            text=f"Progress: {self.bench_genuine_count}/{self.bench_target_genuine} genuine, {self.bench_impostor_count}/{self.bench_target_impostor} impostor")

        if self.bench_genuine_count >= self.bench_target_genuine:
            self.mode = "bench_impostor"
            self._log("→ Now do impostor tests", "yellow")
            self.root.after(1500, self._build_benchmark_screen)

    def _handle_benchmark_impostor(self, features):
        score = self.engine.score(features)
        threshold = THRESHOLD_AUTH if self.pm.compute_phase(self.profile) == "adaptation" else THRESHOLD_GROWTH
        accepted = score <= threshold

        imp_label = f"imp_{self.bench_impostor_count+1}"
        self.recorder.record_impostor(score, label=imp_label)
        self.bench_impostor_features.append(features)
        self.bench_impostor_scores.append(float(score))
        self.bench_impostor_count += 1
        self._record_to_dataset("bench_impostor", self.capturer.events, accepted=accepted, label=imp_label)

        # For impostors: accepted = FAR (bad)
        if accepted:
            self.status_dot.config(text=f"⚠ ACCEPTED  score={score:.3f}  (false accept — FAR)", fg=RED)
            self._log(f"  [impostor {self.bench_impostor_count}] ⚠ SLIPPED IN  score={score:.3f}  [FAR]", "red")
        else:
            self.status_dot.config(text=f"✅ REJECTED  score={score:.3f}  (correct — impostor)", fg=GREEN)
            self._log(f"  [impostor {self.bench_impostor_count}] ✅ REJECTED  score={score:.3f}", "green")

        self.counter_label.config(
            text=f"Progress: {self.bench_genuine_count}/{self.bench_target_genuine} genuine, {self.bench_impostor_count}/{self.bench_target_impostor} impostor")

        if self.bench_impostor_count >= self.bench_target_impostor:
            self.root.after(1500, self._show_benchmark_results)

    def _show_benchmark_results(self):
        """Display results + save summary to profile history."""
        from datetime import datetime
        assert self.profile is not None

        phase_info = self.pm.phase_info(self.profile)
        threshold  = THRESHOLD_AUTH if phase_info["phase"] == "adaptation" else THRESHOLD_GROWTH
        far        = self.recorder.compute_far(threshold)
        frr        = self.recorder.compute_frr(threshold)
        eer, eer_t = self.recorder.compute_eer()

        # Save to profile as a benchmark snapshot entry (with raw data for aggregation)
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

        # Build results screen
        for w in self.root.winfo_children():
            w.destroy()

        self._build_header("BENCHMARK RESULTS",
                           f"Profile maturity: {phase_info['sample_count']} samples • {phase_info['phase']} phase",
                           YELLOW)

        panel = self._build_main_panel()

        # Big metrics row
        metrics = tk.Frame(panel, bg=PANEL)
        metrics.pack(fill="x", padx=30, pady=(20, 10))
        for label, val, color in [
            ("FAR", f"{far*100:.1f}%", RED if far > 0.2 else GREEN),
            ("FRR", f"{frr*100:.1f}%", RED if frr > 0.2 else GREEN),
            ("EER", f"{eer*100:.1f}%" if eer is not None else "—",
                     YELLOW if eer and eer > 0.10 else GREEN),
            ("THRESHOLD", f"{threshold}", TEXT),
        ]:
            cell = tk.Frame(metrics, bg="#0f1117",
                            highlightbackground=BORDER, highlightthickness=1)
            cell.pack(side="left", expand=True, fill="both", padx=4)
            tk.Label(cell, text=label, bg="#0f1117", fg=TEXT_DIM,
                     font=("Courier", 8, "bold")).pack(pady=(10, 2))
            tk.Label(cell, text=val, bg="#0f1117", fg=color,
                     font=("Courier", 18, "bold")).pack(pady=(0, 10))

        # Interpretation
        if eer is not None:
            if eer < 0.10:
                interp, col = "✓ Excellent — ready for production", GREEN
            elif eer < 0.20:
                interp, col = "~ Acceptable — needs more maturity", YELLOW
            else:
                interp, col = "! Needs more samples — keep using it", RED
            tk.Label(panel, text=interp, bg=PANEL, fg=col,
                     font=("Courier", 11, "bold")).pack(pady=10)

        # Benchmark history (trend tracking)
        history = self.profile.get("benchmark_history", [])
        if len(history) > 1:
            tk.Label(panel, text="📈 BENCHMARK HISTORY (profile maturing)",
                     bg=PANEL, fg=TEXT_DIM,
                     font=("Courier", 9, "bold")).pack(anchor="w", padx=30, pady=(15, 4))
            hist_frame = tk.Frame(panel, bg="#0f1117",
                                  highlightbackground=BORDER, highlightthickness=1)
            hist_frame.pack(fill="x", padx=30, pady=(0, 10))

            header = "  Run  │ Samples │ Phase      │ EER    │ FAR    │ FRR"
            tk.Label(hist_frame, text=header, bg="#0f1117", fg=TEXT_DIM,
                     font=("Courier", 8, "bold"),
                     justify="left", anchor="w").pack(fill="x", padx=8, pady=(6, 2))

            for i, h in enumerate(history[-6:], start=max(1, len(history)-5)):
                eer_str = f"{h['eer']*100:5.1f}%" if h.get("eer") is not None else "  —  "
                row = f"   {i:3} │  {h['sample_count']:5}  │ {h['phase']:10} │ {eer_str} │ {h['far']*100:5.1f}% │ {h['frr']*100:5.1f}%"
                color = GREEN if i == len(history) else TEXT_DIM
                tk.Label(hist_frame, text=row, bg="#0f1117", fg=color,
                         font=("Courier", 8),
                         justify="left", anchor="w").pack(fill="x", padx=8)
            tk.Frame(hist_frame, bg="#0f1117", height=4).pack()

        # Actions
        btn_row = tk.Frame(panel, bg=PANEL)
        btn_row.pack(pady=20)
        tk.Button(btn_row, text="📊 VIEW CHARTS",
                  bg=ACCENT, fg=TEXT_BRIGHT,
                  font=("Courier", 10, "bold"),
                  relief="flat", bd=0, padx=20, pady=10, cursor="hand2",
                  command=self._show_benchmark_charts).pack(side="left", padx=6)
        tk.Button(btn_row, text="← BACK TO AUTH",
                  bg=BORDER, fg=TEXT_DIM,
                  font=("Courier", 10, "bold"),
                  relief="flat", bd=0, padx=20, pady=10, cursor="hand2",
                  command=self._exit_benchmark).pack(side="left", padx=6)

        self._build_log_panel()
        self._log("═══ RESULTS ═══", "yellow")
        self._log(f"FAR: {far*100:.1f}%", "red" if far > 0.2 else "green")
        self._log(f"FRR: {frr*100:.1f}%", "red" if frr > 0.2 else "green")
        if eer is not None:
            self._log(f"EER: {eer*100:.1f}% @ t={eer_t:.3f}", "yellow")
        self._log(f"Saved to profile history", "dim")

    def _show_benchmark_charts(self):
        import os
        from datetime import datetime

        # Auto-save path inside profiles/<user>_charts/
        charts_dir = str(_HERE / "profiles" / f"{self.username}_charts")
        stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_ct  = self.profile.get("sample_count", 0)
        filename   = f"benchmark_{stamp}_samples{sample_ct}.png"
        save_path  = os.path.join(charts_dir, filename)

        viz = BenchmarkVisualizer(
            recorder=self.recorder,
            engine=self.engine,
            genuine_features=self.bench_genuine_features,
            impostor_features=self.bench_impostor_features,
            threshold=THRESHOLD_AUTH if self.pm.compute_phase(self.profile) == "adaptation" else THRESHOLD_GROWTH,
            benchmark_history=self.profile.get("benchmark_history", []),
            save_path=save_path
        )
        viz.show_all()
        self._log(f"📸 Chart saved: {filename}", "accent")

    def _exit_benchmark(self):
        self.mode = "auth"
        self.bench_frozen_profile = None
        self._build_auth_screen()

    # ═════════════════════════════════════════════════════
    # NAVIGATION
    # ═════════════════════════════════════════════════════
    def _logout(self):
        self.username        = None
        self.profile         = None
        self.mode            = None
        self.enroll_count    = 0
        self.enroll_features = []
        self._build_login_screen()

    # ═════════════════════════════════════════════════════
    # DEV MENU (Ctrl+Shift+D)
    # ═════════════════════════════════════════════════════
    def _open_dev_menu(self, event=None):
        dev = tk.Toplevel(self.root)
        dev.title("KLAVIS — Dev Menu")
        dev.configure(bg=BG)
        dev.geometry("520x640")
        dev.transient(self.root)

        tk.Label(dev, text="⚠ DEV MENU",
                 bg=BG, fg=YELLOW,
                 font=("Courier", 14, "bold")).pack(pady=20)

        tk.Label(dev, text="Profiles on disk:",
                 bg=BG, fg=TEXT,
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=20, pady=(10, 4))

        users = self.pm.list_users()
        list_frame = tk.Frame(dev, bg=PANEL,
                              highlightbackground=BORDER, highlightthickness=1)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        if not users:
            tk.Label(list_frame, text="(no profiles)",
                     bg=PANEL, fg=TEXT_DIM,
                     font=("Courier", 10)).pack(pady=20)
        else:
            for user in users:
                row = tk.Frame(list_frame, bg=PANEL)
                row.pack(fill="x", padx=8, pady=2)
                profile = self.pm.load(user)
                info    = self.pm.phase_info(profile)
                label = f"{user}  •  {info['sample_count']} samples  •  {info['phase']}"
                tk.Label(row, text=label, bg=PANEL, fg=TEXT,
                         font=("Courier", 9)).pack(side="left", padx=8)
                tk.Button(row, text="DELETE",
                          bg=RED, fg=TEXT_BRIGHT,
                          font=("Courier", 8, "bold"),
                          relief="flat", bd=0, padx=10, pady=3, cursor="hand2",
                          command=lambda u=user, d=dev: self._dev_delete(u, d)).pack(side="right", padx=6)

        # Dataset section
        tk.Frame(dev, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)
        tk.Label(dev, text="DATASET (CMU-schema CSV)",
                 bg=BG, fg=ACCENT2,
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=20, pady=(4, 6))

        stats = self.dm.stats()
        stats_text = f"  {len(stats['users'])} users consented  •  {stats['total_rows']} rows total"
        tk.Label(dev, text=stats_text, bg=BG, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(anchor="w", padx=20)

        ds_btn_row = tk.Frame(dev, bg=BG)
        ds_btn_row.pack(pady=10)
        tk.Button(ds_btn_row, text="📦 EXPORT MASTER CSV",
                  bg=ACCENT, fg=TEXT_BRIGHT,
                  font=("Courier", 9, "bold"),
                  relief="flat", bd=0, padx=16, pady=8, cursor="hand2",
                  command=lambda: self._dev_export_dataset(dev)).pack(side="left", padx=4)
        tk.Button(ds_btn_row, text="📂 OPEN DATASETS FOLDER",
                  bg=BORDER, fg=TEXT,
                  font=("Courier", 9, "bold"),
                  relief="flat", bd=0, padx=16, pady=8, cursor="hand2",
                  command=self._dev_open_datasets).pack(side="left", padx=4)

        # Nuclear option
        tk.Frame(dev, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)
        tk.Button(dev, text="💥 DELETE ALL PROFILES",
                  bg=RED, fg=TEXT_BRIGHT,
                  font=("Courier", 10, "bold"),
                  relief="flat", bd=0, padx=20, pady=8, cursor="hand2",
                  command=lambda: self._dev_delete_all(dev)).pack(pady=6)

        tk.Button(dev, text="CLOSE",
                  bg=BORDER, fg=TEXT_DIM,
                  font=("Courier", 9, "bold"),
                  relief="flat", bd=0, padx=16, pady=6, cursor="hand2",
                  command=dev.destroy).pack(pady=(6, 10))

    def _dev_export_dataset(self, dev_window):
        path, rows = self.dm.export_master_dataset()
        if path is None:
            messagebox.showwarning("Empty", "No dataset files to export yet.")
            return
        messagebox.showinfo("Exported",
                           f"Merged {rows} rows into:\n\n{path}")

    def _dev_open_datasets(self):
        import subprocess
        from dataset_manager import DATASETS_DIR
        path = os.path.abspath(DATASETS_DIR)
        if sys.platform == "darwin":
            subprocess.run(["open", path])
        elif sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.run(["xdg-open", path])

    def _dev_delete(self, username, dev_window):
        if messagebox.askyesno("Confirm", f"Delete profile for {username}?"):
            self.pm.delete(username)
            dev_window.destroy()
            self._open_dev_menu()  # refresh

    def _dev_delete_all(self, dev_window):
        if messagebox.askyesno("Confirm",
                               "Delete ALL profiles? This cannot be undone."):
            count = self.pm.delete_all()
            messagebox.showinfo("Done", f"Deleted {count} profiles.")
            dev_window.destroy()
            if self.mode is not None:
                self._logout()


def main():
    root = tk.Tk()
    root.lift()
    root.attributes("-topmost", True)
    root.after(100, lambda: root.attributes("-topmost", False))
    root.focus_force()
    app = KlavisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()