"""
model_core.py — shared constants and pure utility functions
===========================================================
Imported by both march_madness_2026_v4.py (bracket optimizer)
and backtest/backtest_harness_v6.py (backtesting harness).

Placing shared logic here ensures both pipelines stay in sync
and eliminates drift between identically-named functions.

Functions kept here: pure math / transform / lookup only.
All I/O, calibration, simulation, and reporting stay in their
respective files.
"""

import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t

# =============================================================================
# CONSTANTS
# =============================================================================

CORPUS_INJURY_BETA = -0.244   # locked from 2021-2024 empirical corpus

# Round-specific sigmoid temperature — compresses extreme WP in later rounds
# (surviving teams are more evenly matched; same logit advantage → less certainty)
ROUND_TEMP = {
    'R1':    1.00,
    'R2':    0.92,
    'S16':   0.84,
    'E8':    0.76,
    'FF':    0.70,
    'Champ': 0.70,
}

# 35-year historical R1 win rates by seed matchup (higher seed winning)
SEED_PRIORS = {
    (1, 16): 0.994,
    (2, 15): 0.943,
    (3, 14): 0.854,
    (4, 13): 0.796,
    (5, 12): 0.649,
    (6, 11): 0.629,
    (7, 10): 0.605,
    (8,  9): 0.505,
}

# Model weight in Bayesian blend: higher α → model dominates; lower → prior dominates
# Close matchups (8v9) use more model; blowouts (1v16) defer to 35-year prior
SEED_ALPHA = {
    (8,  9): 0.85,
    (7, 10): 0.78,
    (6, 11): 0.72,
    (5, 12): 0.70,
    (4, 13): 0.60,
    (3, 14): 0.55,
    (2, 15): 0.40,
    (1, 16): 0.30,
}

# Conference champion sets — power vs mid-major split
POWER_CONFS = {'ACC', 'Big Ten', 'Big 12', 'SEC', 'Big East', 'Pac-12', 'Pac-10'}

POWER_CHAMPS = {
    2019: {'Virginia', 'Texas Tech'},
    2024: {'Iowa State', 'Houston', 'Purdue', 'North Carolina', 'Duke', 'Arizona'},
    2025: {'Florida', 'Auburn', "St. John's", 'Purdue', 'Arizona'},
    2026: {'Duke', 'Purdue', "St. John's", 'Arizona', 'Arkansas'},
}

MID_CHAMPS = {
    2019: {'Gonzaga', 'UC Irvine', 'Liberty', 'Old Dominion', 'Gardner Webb',
           'Iona', 'Montana', 'Buffalo', 'Murray State', 'Wofford',
           'North Dakota State', 'Fairleigh Dickinson', 'Abilene Christian',
           'Colgate', 'Georgia State', 'UNCG', 'Bradley', 'UC San Diego'},
    2024: {'Gonzaga', 'James Madison', 'Oakland', 'UAB', 'Duquesne',
           'New Mexico', 'South Carolina', 'South Dakota St', 'Utah State'},
    2025: {'Drake', 'McNeese', 'Gonzaga'},
    2026: {'Gonzaga', 'Utah State', 'N. Iowa', 'S. Florida', 'Akron',
           'Howard', 'Lehigh', 'Hofstra', 'Cal Baptist', 'Queens',
           'McNeese', 'Idaho', 'Tenn. State', 'ND State', 'Wright State',
           'Penn', 'Siena', 'LIU', 'Furman', 'Troy', 'Kennesaw St.', 'Pr. View'},
}

# =============================================================================
# LOG TRANSFORMS
# =============================================================================

def log_norm(x: float, scale: float = 10.0) -> float:
    """Compress x into [0,1] via log1p, normalised to scale."""
    return np.log1p(max(0.0, float(x))) / np.log1p(scale)


def log_rank(r: float, norm: float = 80.0) -> float:
    """Log-compress a KenPom rank (lower rank = stronger team)."""
    return np.log1p(max(0.0, float(r))) / np.log1p(norm)

# =============================================================================
# BAYESIAN SEED-MATCHUP PRIOR BLEND
# =============================================================================

def get_seed_prior(seed_a, seed_b):
    """
    Return (prior_wp_a, alpha) for a given R1 matchup, or (None, None).
    Normalises so the lower seed is always the 'home' key in SEED_PRIORS.
    """
    if seed_a is None or seed_b is None:
        return None, None
    sa, sb = int(seed_a), int(seed_b)
    flipped = sa > sb
    if flipped:
        sa, sb = sb, sa
    key = (sa, sb)
    if key not in SEED_PRIORS:
        return None, None
    prior = SEED_PRIORS[key] if not flipped else 1.0 - SEED_PRIORS[key]
    alpha = SEED_ALPHA.get(key, 0.65)
    return prior, alpha


def bayesian_blend(p_model: float, seed_a, seed_b) -> float:
    """
    Blend model win-probability with 35-year empirical R1 seed-matchup prior.

    p_final = alpha * p_model + (1 - alpha) * p_prior

    alpha varies by matchup: 0.30 (1v16, prior dominates) → 0.85 (8v9).
    Returns p_model unchanged if no prior exists for this matchup.
    """
    prior, alpha = get_seed_prior(seed_a, seed_b)
    if prior is None:
        return p_model
    return alpha * p_model + (1.0 - alpha) * prior

# =============================================================================
# SIGMOID WITH ROUND-SPECIFIC TEMPERATURE
# =============================================================================

def tempered_sig(logit: float, rnd: str) -> float:
    """
    Sigmoid with round-specific temperature compression.
    Later rounds compress extreme probabilities: same logit → less certainty
    because surviving teams are more evenly matched.
    """
    T = ROUND_TEMP.get(rnd, 1.0)
    return float(expit(logit / T))

# =============================================================================
# STUDENT-t CHAOS EPSILON
# =============================================================================

def chaos_eps(rng, df: int = 4, scale: float = 0.18) -> float:
    """
    Draw a single chaos perturbation from a Student-t distribution.
    df=4 gives 2.6x more probability mass in tails vs Normal(0, scale),
    matching historical frequency of high-upset tournaments (2022 style).
    """
    return float(student_t.rvs(df=df, scale=scale, random_state=rng))

# =============================================================================
# INJURY RESIDUAL
# =============================================================================

def vegas_coverage(sev: float, sharp: float) -> float:
    """Fraction of injury severity already priced into Vegas line."""
    return max(min(1.0, sev / 2.5), min(1.0, abs(sharp) / 2.0))


def injury_residual(sev: float, sharp: float) -> float:
    """Injury severity not yet reflected in the Vegas line."""
    return sev * (1.0 - vegas_coverage(sev, sharp))

# =============================================================================
# CONFERENCE CHAMPION FLAGS
# =============================================================================

def get_champ_flags(team: str, year: int):
    """Return (power_champ_float, mid_champ_float) for a team/year."""
    pc = float(team in POWER_CHAMPS.get(year, set()))
    mc = float(team in MID_CHAMPS.get(year, set()))
    return pc, mc
