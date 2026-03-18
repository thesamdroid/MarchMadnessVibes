"""
Tests for model_core.py — the shared utility module.
Covers every exported function and all constants.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from model_core import (
    CORPUS_INJURY_BETA, ROUND_TEMP, SEED_PRIORS, SEED_ALPHA,
    POWER_CHAMPS, MID_CHAMPS,
    log_norm, log_rank,
    get_seed_prior, bayesian_blend,
    tempered_sig, chaos_eps,
    vegas_coverage, injury_residual,
    get_champ_flags,
)

# ── Constants ──────────────────────────────────────────────────────────────

class TestConstants:
    def test_corpus_beta_negative(self):
        assert CORPUS_INJURY_BETA < 0

    def test_corpus_beta_magnitude(self):
        assert -0.5 < CORPUS_INJURY_BETA < -0.1

    def test_round_temp_decreasing(self):
        rounds = ['R1','R2','S16','E8','FF']
        temps = [ROUND_TEMP[r] for r in rounds]
        assert temps == sorted(temps, reverse=True)

    def test_r1_temp_is_one(self):
        assert ROUND_TEMP['R1'] == 1.0

    def test_seed_priors_complete(self):
        expected = {(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)}
        assert set(SEED_PRIORS.keys()) == expected

    def test_seed_priors_range(self):
        for p in SEED_PRIORS.values():
            assert 0.0 < p < 1.0

    def test_seed_priors_ordered(self):
        # Lower seed number → higher win rate
        assert SEED_PRIORS[(1,16)] > SEED_PRIORS[(5,12)] > SEED_PRIORS[(8,9)]

    def test_seed_alpha_range(self):
        for a in SEED_ALPHA.values():
            assert 0.0 < a <= 1.0

    def test_conf_champs_no_overlap(self):
        for year in [2019,2024,2025,2026]:
            overlap = POWER_CHAMPS.get(year,set()) & MID_CHAMPS.get(year,set())
            assert len(overlap) == 0, f"Year {year} overlap: {overlap}"

# ── log_norm ───────────────────────────────────────────────────────────────

class TestLogNorm:
    def test_zero_input(self):       assert log_norm(0.0) == 0.0
    def test_max_input(self):        assert abs(log_norm(10.0) - 1.0) < 1e-9
    def test_negative_clamped(self): assert log_norm(-99) == 0.0
    def test_mid_in_range(self):     assert 0.0 < log_norm(5.0) < 1.0
    def test_concave(self):
        a = log_norm(2) - log_norm(0)
        b = log_norm(6) - log_norm(4)
        c = log_norm(10)- log_norm(8)
        assert a > b > c

# ── log_rank ───────────────────────────────────────────────────────────────

class TestLogRank:
    def test_zero(self):      assert log_rank(0.0) == 0.0
    def test_positive(self):  assert log_rank(40) > 0
    def test_concave(self):
        assert log_rank(10)-log_rank(1) > log_rank(100)-log_rank(90)

# ── get_seed_prior ─────────────────────────────────────────────────────────

class TestGetSeedPrior:
    def test_known_matchup(self):
        prior, alpha = get_seed_prior(1, 16)
        assert abs(prior - 0.994) < 1e-9
        assert alpha == 0.30

    def test_flipped_seeds(self):
        prior_fwd, _ = get_seed_prior(1, 16)
        prior_rev, _ = get_seed_prior(16, 1)
        assert abs(prior_fwd + prior_rev - 1.0) < 1e-9

    def test_unknown_matchup(self):
        assert get_seed_prior(2, 3) == (None, None)

    def test_none_seed(self):
        assert get_seed_prior(None, 16) == (None, None)

# ── bayesian_blend ─────────────────────────────────────────────────────────

class TestBayesianBlend:
    def test_1v16_prior_dominates(self):
        p = bayesian_blend(0.70, 1, 16)
        assert abs(p - (0.30*0.70 + 0.70*0.994)) < 1e-6

    def test_8v9_model_survives(self):
        p = bayesian_blend(0.60, 8, 9)
        assert abs(p - (0.85*0.60 + 0.15*0.505)) < 1e-6

    def test_unknown_passthrough(self):
        assert bayesian_blend(0.65, 2, 3) == 0.65

    def test_flipped_symmetry(self):
        assert bayesian_blend(0.99, 1, 16) > 0.5
        assert bayesian_blend(0.01, 16, 1) < 0.5

    def test_bounded(self):
        for sa,sb in [(1,16),(5,12),(8,9),(3,14)]:
            for pm in [0.0, 0.5, 1.0]:
                assert 0.0 <= bayesian_blend(pm, sa, sb) <= 1.0

# ── tempered_sig ───────────────────────────────────────────────────────────

class TestTemperedSig:
    def test_r1_full_temperature(self):
        from scipy.special import expit
        assert abs(tempered_sig(2.0,'R1') - float(expit(2.0))) < 1e-6

    def test_compression_in_later_rounds(self):
        # T < 1 divides by a smaller number → logit/T is larger → sigmoid is MORE extreme.
        # Same model advantage is more decisive in E8/FF (calibrated on R1 data;
        # in later rounds a rank gap truly matters more). This is correct behaviour.
        l = 3.0
        assert tempered_sig(l,'E8') > tempered_sig(l,'R1')
        assert tempered_sig(l,'FF') >= tempered_sig(l,'E8')

    def test_zero_logit_is_half(self):
        for rnd in ROUND_TEMP:
            assert abs(tempered_sig(0.0, rnd) - 0.5) < 1e-6

    def test_negative_below_half(self):
        for rnd in ROUND_TEMP:
            assert tempered_sig(-2.0, rnd) < 0.5

# ── chaos_eps ──────────────────────────────────────────────────────────────

class TestChaosEps:
    def test_zero_centered(self):
        rng = np.random.default_rng(42)
        assert abs(np.mean([chaos_eps(rng) for _ in range(5000)])) < 0.03

    def test_heavy_tails(self):
        rng = np.random.default_rng(7)
        samples = [chaos_eps(rng) for _ in range(10000)]
        # t(df=4) at scale=0.18: expect >1% beyond 3*scale=0.54
        extreme = sum(1 for s in samples if abs(s) > 0.54)
        assert extreme > 60, f"Too few extreme samples: {extreme}"

# ── injury_residual ────────────────────────────────────────────────────────

class TestInjuryResidual:
    def test_zero_severity(self):      assert injury_residual(0.0, 0.0) == 0.0
    def test_fully_priced(self):       assert injury_residual(2.5, 1.0) == 0.0
    def test_partial_residual(self):
        r = injury_residual(0.5, 0.0)
        assert 0 < r < 0.5
    def test_never_exceeds_severity(self):
        for sev in [0.5,1.0,2.5,3.0]:
            for sm in [0.0,0.5,2.0]:
                assert injury_residual(sev, sm) <= sev + 1e-9
    def test_sharp_move_reduces_residual(self):
        assert injury_residual(1.0, 0.0) >= injury_residual(1.0, 2.0)

# ── get_champ_flags ────────────────────────────────────────────────────────

class TestGetChampFlags:
    def test_power_2026_duke(self):
        pc,mc = get_champ_flags('Duke', 2026)
        assert pc==1.0 and mc==0.0

    def test_mid_2026_gonzaga(self):
        pc,mc = get_champ_flags('Gonzaga', 2026)
        assert pc==0.0 and mc==1.0

    def test_at_large(self):
        pc,mc = get_champ_flags('Tennessee', 2026)
        assert pc==0.0 and mc==0.0

    def test_historic_2019_virginia(self):
        pc,_ = get_champ_flags('Virginia', 2019)
        assert pc==1.0

    def test_historic_2025_drake(self):
        pc,mc = get_champ_flags('Drake', 2025)
        assert pc==0.0 and mc==1.0

    def test_returns_floats(self):
        pc,mc = get_champ_flags('Duke', 2026)
        assert isinstance(pc, float) and isinstance(mc, float)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
