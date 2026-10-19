"""All tunable settings in one place. Values marked ASSUMED are not from data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- L1 solar wind, 1-minute (CDAWeb HAPI). OMNI is gap-filled across ACE/Wind/DSCOVR,
# which matters for a long ML training set. NOTE: OMNI is time-shifted to the bow shock,
# so it is NOT used for timing the L1 contact — only for speeds, densities and fields.
L1_DATASET = "OMNI_HRO_1MIN"
L1_PARAMS = {"V": "flow_speed", "N": "proton_density", "BZ": "BZ_GSM", "B": "F"}

# ---- High-resolution magnetometer for structure-size estimates (tried in order).
HIRES_MAG = [
    ("DSCOVR_H0_MAG", "B1GSE"),   # 1-s DSCOVR
    ("WI_H0_MFI@1", "B3GSE"),     # 3-s Wind (CDAWeb split WI_H0_MFI into @0/@1/@2)
]

# ---- Swarm geometry (km offsets from L1: x sunward-negative, y, z) — same as the UI.
SAT = {
    "S1": (-750e3, 0, 0),
    "S2": (-20e3, -240e3, -60e3),
    "S3": (-20e3, 240e3, -60e3),
    "S4": (30e3, 0, 250e3),
    "S5": (-20e3, -237e3, -60e3),
}
SUB_L1_OFFSET_KM = 750e3

# ---- DBM literature default (used as the "M0" baseline).
GAMMA0 = 2e-8          # 1/km
W0 = 400.0             # km/s

# ---- Scout optimizer (UI "Optimization" stage).
SCOUT = dict(
    noise_ratio=0.075,          # ASSUMED: 0.3 nT noise on a 4 nT signal
    L0=3000.0,                  # current S2-S5 separation, km
    budget_ms=40.0,             # ASSUMED delta-v budget per event, m/s
    window_h=40.0,              # time available before contact, h
    u_threshold=0.25,           # move only if expected error exceeds this
    candidates=[1000, 1400, 1800, 2400, 3000, 3600, 4000, 4400],
    weights=[0.001, 0.002, 0.004, 0.006, 0.01],   # sensitivity sweep on the Δv weight
    lam_prior_km=9000.0, lam_sigma_ln=0.5,       # ASSUMED fallback if no data run yet
)

SPLIT_FRACTION = 0.7    # chronological train/test split for DBM validation
SHOCK_TRAIN_END = "2021-12-31"   # shock detector: train <= this date, test after
