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
assert bayesian_blend(0.70, 1, 16) < 0.70, "Bayesian blend should pull 1v16 toward strong prior"
assert abs(tempered_sig(0.0, "R1") - 0.5) < 1e-6, "Zero logit must be 50% at all temps"

print(f"P(perfect R1):    {chalk['jp_r1p']*100:.5f}%")
print(f"P(perfect R1+R2): {chalk['jp_r1r2p']*100:.6f}%")
print("All smoke tests passed.")
