#!/usr/bin/env python3
"""CI smoke test — run by GitHub Actions workflow."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from model_core import CORPUS_INJURY_BETA, bayesian_blend, tempered_sig
from march_madness_2026_v4 import (
    load_data, resolve_bracket, build_chalk, calibrate, FIRST_FOUR_RESULTS
)
import numpy as np

teams, vegas = load_data()
bracket = resolve_bracket(FIRST_FOUR_RESULTS)
beta = calibrate(vegas, teams, n_steps=500, verbose=False)
chalk = build_chalk(beta, bracket, teams, protect=True)

assert 0 < chalk["jp_r1p"] < 1, f"P(R1) out of range: {chalk['jp_r1p']}"
assert chalk["jp_r1r2p"] < chalk["jp_r1p"], "R1+R2 must be strictly less than R1"
assert CORPUS_INJURY_BETA == -0.244, f"Injury beta must be locked at -0.244, got {CORPUS_INJURY_BETA}"

# Bayesian blend: model undershooting the 99.4% 1v16 prior should be pulled UP
p_blend = bayesian_blend(0.40, 1, 16)
assert p_blend > 0.40, f"Blend should pull undershooting model UP toward prior, got {p_blend:.4f}"

# Zero logit -> exactly 50% at any temperature
assert abs(tempered_sig(0.0, "R1") - 0.5) < 1e-6, "Zero logit must be 50% at all temps"

# R2 same-logit should be more extreme than R1 (T<1 amplifies)
assert tempered_sig(2.0, "R2") > tempered_sig(2.0, "R1"), "Lower T should amplify positive logit"

print(f"P(perfect R1):    {chalk['jp_r1p']*100:.5f}%")
print(f"P(perfect R1+R2): {chalk['jp_r1r2p']*100:.6f}%")
print(f"Bayesian blend(0.40, 1v16): {p_blend:.4f}")
print("All smoke tests passed.")
