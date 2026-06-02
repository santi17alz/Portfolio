# KLAVIS — Keystroke Biometric Hardware Token

> A two-factor authentication system that integrates adaptive keystroke biometrics into the FIDO2 hardware token flow — verifying not just *what you have*, but *who you are*.

**ECE 547/647 Security Engineering · University of Massachusetts Amherst · May 2026**  
*Santiago Alzate · Kyle Belanger · Tomer Carmeli · Jeffrey Kidwell*

---

## The Problem

FIDO2 hardware tokens provide strong cryptographic guarantees — but they operate under a critical assumption: **possession of the token equals proof of identity**. If the token is stolen or borrowed, that assumption breaks. KLAVIS adds a behavioral biometric gate — **your typing rhythm** — directly in front of the WebAuthn flow. The token only proceeds with FIDO2 challenge-response after the person holding it proves they type like the enrolled owner.

---

## System Architecture

```
Solo2 USB inserted
        │
        ▼
  USB Daemon (usb_find.py)          ← cross-platform: WMI / pyudev / ioreg
        │
        ├── Certificate expired? ──► Re-enrollment GUI
        │
        └── Valid cert ──► User GUI  or  Developer GUI
                                │
                                ▼
                     Keystroke Capture (pynput)
                                │
                                ▼
                     Feature Extraction (features.py)
                     dwell time · flight time · digraph latency
                                │
                                ▼
                     Manhattan Scaled Engine (engine.py)
                     Bloom filter → MAD-normalized scoring → threshold
                                │
                         Accept / Reject
                                │
                     Profile Manager (profile_manager.py)
                     Growth phase (EWA)  or  Adaptation phase (EMA)
```

---

## Core Algorithm: Scaled Manhattan Distance + MAD Normalization

Based on the top-performing algorithm from the Killourhy & Maxion (2009) CMU benchmark.

**Enrollment** builds a per-user profile:
1. Raw keystroke events → bigram feature vectors `[dwell, flight]` per key pair
2. IQR outlier filtering removes anomalous enrollment attempts
3. Per-feature **mean** and **mean absolute deviation (MAD)** computed across cleaned samples
4. Bigrams registered in a **Bloom filter** for fast pre-screening

**Authentication** runs three stages:
1. **Bloom filter check** — rejects attempts with bigrams never seen in enrollment (coverage threshold: 70%)
2. **Scaled Manhattan score** — each feature's deviation from the enrolled mean is divided by its MAD, then averaged. Naturally variable features contribute less; stable features contribute more.
3. **Threshold comparison** — score ≤ threshold → accept; above → reject

```
score = mean( |live_feature − enrolled_mean| / enrolled_MAD )
```

The score is fully interpretable and works reliably with as few as 8–20 enrollment samples, unlike SVM or neural approaches that require hundreds.

---

## Key Engineering Decisions

### Adaptive Enrollment (8–20 samples)
Enrollment stops early when the profile is statistically stable, evaluated over a sliding window using two signals:
- **Median coefficient of variation (CV)** across active bigram features — target ≤ 0.10
- **Average self-score** of recent samples against the current profile — target ≤ 0.80

Consistent typists finish in 8–12 samples. Inconsistent typists reach the 20-sample maximum.

### Two-Phase Profile Lifecycle

| Phase | Trigger | Update Rule |
|---|---|---|
| **Growth** | Post-enrollment | Equal-weight rolling mean: `(mean × N + new) / (N + 1)` |
| **Adaptation** | Stability criteria met | Exponential moving average: `(1 − α) × old + α × new`, α = 0.05 |

The adaptation phase lets the profile slowly track natural drift in typing behavior over months without overreacting to individual attempts.

### Dual Threshold Design (anti-poisoning)

| Threshold | Value | Purpose |
|---|---|---|
| Authentication | 2.5 (growth) / 3.0 (adaptation) | Grants or denies access |
| Learning | 1.5 | Determines if the attempt updates the stored profile |

Borderline accepted attempts (score between 1.5 and the auth threshold) grant access but **do not update the profile**. This prevents a successful attacker from gradually poisoning the biometric template toward their own typing pattern — a vulnerability discovered during testing when using a single threshold for both decisions.

### 90-Day Certificate System
The USB daemon tracks when a user's profile was last enrolled. After 90 days, the user is prompted to re-enroll, accounting for long-term drift in typing patterns.

---

## Results

| Metric | Value |
|---|---|
| Best single-run FAR / FRR / EER | **0% / 0% / 0%** |
| Pooled EER (11 runs · 20 genuine · 30 impostor) | **0.00%** @ threshold 4.70 |
| EER improvement per additional sample | ~0.37–0.38 percentage points |
| Sample range for consistent 0% EER | 35–70 profile samples |

Early benchmark runs on immature profiles showed EER values of 20–40% — consistent with the cold-start design. Later runs on mature profiles consistently achieved 0% error across all three metrics.

---

## File Structure

```
klavis/
├── engine.py            # Manhattan Scaled engine + Bloom filter
├── features.py          # Feature extraction: dwell, flight, bigram filtering
├── capture.py           # Keystroke event accumulator
├── profile_manager.py   # Profile persistence, growth/adaptation lifecycle, EMA
├── adaptive_policy.py   # Enrollment stability checker (CV + self-score)
├── benchmark.py         # FAR / FRR / EER computation
├── aggregator.py        # Cross-run and cross-user benchmark aggregation
├── dataset_manager.py   # CMU-schema CSV export for research
├── usb_find.py          # USB daemon: Solo2 detection, certificate check, GUI launch
├── user_gui.py          # End-user GUI (clean UX, hides scoring internals)
├── developer_gui.py     # Developer GUI (live logs, scores, benchmark, plots)
├── visualize.py         # Per-run visualization dashboard (6-panel)
├── aggregate_viz.py     # Cross-run aggregated benchmark plots
└── KLAVIS_Final_Report.pdf
```

---

## Setup

**Requirements:** Python 3.9+, SoloKeys Solo2 USB token

```bash
git clone https://github.com/your-username/klavis   # TODO: update URL
cd klavis
pip install pynput numpy pyudev wmi                 # TODO: add requirements.txt
```

**Run — user mode:**
```bash
python usb_find.py
# Plug in your Solo2; the GUI launches automatically
```

**Run — developer mode:**
```bash
python usb_find.py developer
```

> Pre-built executables (.exe / Linux binary) are available under Releases.

---

## Tech Stack

`Python` · `pynput` · `tkinter` · `NumPy` · `WMI (Windows)` · `pyudev (Linux)` · `SoloKeys Solo2 FIDO2/U2F`

---

## References

1. Killourhy & Maxion (2009) — CMU keystroke dynamics benchmark dataset and evaluation
2. Cockell & Halak (2019) — Biometric authentication using keystroke dynamics on hardware tokens. *arXiv:1909.10841*
3. Shadman et al. (2025) — Keystroke Dynamics: Concepts, Techniques, and Applications. *ACM Computing Surveys*
4. SoloKeys — [solo2-hw hardware schematic](https://github.com/solokeys/solo2-hw)
