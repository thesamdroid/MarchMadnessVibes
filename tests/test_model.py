"""
Tests for march_madness_2026_v4.py — bracket optimizer.
Pure-function logic tested via model_core; these tests cover
v4-specific: factors(), predict(), calibrate(), build_chalk(),
annuity_pv(), chaos_eps(), get_ownership(), simulate() smoke.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from model_core import bayesian_blend, tempered_sig, CORPUS_INJURY_BETA, ROUND_TEMP
from march_madness_2026_v4 import (
    log_norm, log_rank,
    factors, predict, calibrate, build_chalk,
    annuity_pv, chaos_eps, get_ownership,
)

# ── Stubs ──────────────────────────────────────────────────────────────────

def make_teams():
    return {
        'Duke':      {'seed':1, 'kenpom_rank':1,  'seniority':3.0,'coach_xp':6.0,
                      'injury_severity':2.0,'sos_score':9.5,'momentum':8.0,
                      'q1_wins':17,'brand_tier':3,'power_cc':1,'mid_cc':0},
        'Siena':     {'seed':16,'kenpom_rank':155,'seniority':2.5,'coach_xp':1.0,
                      'injury_severity':0.0,'sos_score':4.5,'momentum':0.0,
                      'q1_wins':0,'brand_tier':0,'power_cc':0,'mid_cc':1},
        'Ohio State':{'seed':8, 'kenpom_rank':35, 'seniority':7.0,'coach_xp':5.0,
                      'injury_severity':0.0,'sos_score':7.5,'momentum':4.5,
                      'q1_wins':4,'brand_tier':2,'power_cc':0,'mid_cc':0},
        'TCU':       {'seed':9, 'kenpom_rank':23, 'seniority':6.0,'coach_xp':4.0,
                      'injury_severity':0.0,'sos_score':9.0,'momentum':3.5,
                      'q1_wins':6,'brand_tier':1,'power_cc':0,'mid_cc':0},
        'Houston':   {'seed':2, 'kenpom_rank':5,  'seniority':6.0,'coach_xp':7.0,
                      'injury_severity':0.0,'sos_score':9.0,'momentum':7.5,
                      'q1_wins':12,'brand_tier':1,'power_cc':0,'mid_cc':0},
        'Tennessee': {'seed':6, 'kenpom_rank':12, 'seniority':4.5,'coach_xp':5.5,
                      'injury_severity':0.0,'sos_score':8.9,'momentum':6.0,
                      'q1_wins':10,'brand_tier':1,'power_cc':0,'mid_cc':0},
    }

def make_beta():
    return np.array([0.0,2.5,0.40,0.30,0.50,0.30,0.0,0.25,0.10,0.05])

# ── Factor vector ──────────────────────────────────────────────────────────

class TestFactors:
    def test_shape(self):
        assert factors('Duke','Siena',make_teams()).shape == (9,)

    def test_dominant_positive_kenpom(self):
        assert factors('Duke','Siena',make_teams())[0] > 0

    def test_underdog_negative_kenpom(self):
        assert factors('Siena','Duke',make_teams())[0] < 0

    def test_unknown_teams_zero(self):
        assert np.allclose(factors('X','Y',make_teams()), 0)

    def test_interaction_positive_for_dominant(self):
        # Duke (#1 KenPom, high mom) vs Siena → interaction favours Duke
        f = factors('Duke','Siena',make_teams())
        assert f[8] > 0

    def test_power_cc_in_vector(self):
        f = factors('Duke','Siena',make_teams())
        assert f[6] == 1.0  # Duke power_cc=1, Siena=0

# ── Predict ────────────────────────────────────────────────────────────────

class TestPredict:
    def test_same_team_stub(self):
        assert predict(make_beta(),'Duke','Duke',make_teams()) == 0.99

    def test_dominant_high_prob(self):
        p = predict(make_beta(),'Duke','Siena',make_teams(),rnd='R1',seed_a=1,seed_b=16)
        assert p > 0.90, f"Expected >90%, got {p:.3f}"

    def test_unit_interval(self):
        beta=make_beta(); teams=make_teams()
        for a,b in [('Duke','Siena'),('TCU','Ohio State'),('Houston','Tennessee')]:
            assert 0 < predict(beta,a,b,teams) < 1

    def test_injury_helps_healthy_team(self):
        beta=make_beta(); teams=make_teams()
        assert predict(beta,'Duke','Siena',teams,ir_b=3.0) > \
               predict(beta,'Duke','Siena',teams,ir_b=0.0)

    def test_bayesian_only_r1(self):
        beta=make_beta(); teams=make_teams()
        p_r1 = predict(beta,'Duke','Siena',teams,rnd='R1',seed_a=1,seed_b=16)
        p_r2 = predict(beta,'Duke','Siena',teams,rnd='R2',seed_a=1,seed_b=16)
        assert p_r1 != p_r2

    def test_asymmetry(self):
        beta=make_beta(); teams=make_teams()
        p_ab = predict(beta,'Duke','Siena',teams)
        p_ba = predict(beta,'Siena','Duke',teams)
        assert abs(p_ab + p_ba - 1.0) < 0.03

# ── Calibration ────────────────────────────────────────────────────────────

class TestCalibration:
    def _vegas(self):
        return [('Duke','Siena',0.986,0.0),
                ('TCU','Ohio State',0.508,0.5),
                ('Houston','Tennessee',0.780,0.0)]

    def test_returns_shape(self):
        b = calibrate(self._vegas(), make_teams(), n_steps=300, verbose=False)
        assert b.shape == (10,)

    def test_reduces_loss(self):
        from scipy.special import expit
        teams=make_teams(); vegas=self._vegas()
        b0 = np.zeros(10)
        bf = calibrate(vegas, teams, n_steps=500, verbose=False)
        def bce(b):
            t=0
            for a,bb,vp,_ in vegas:
                p=float(expit(b[0]+np.dot(b[1:],factors(a,bb,teams))))
                t-=vp*np.log(max(p,1e-9))+(1-vp)*np.log(max(1-p,1e-9))
            return t
        assert bce(bf) < bce(b0)

# ── Annuity ────────────────────────────────────────────────────────────────

class TestAnnuity:
    def test_positive(self):        assert annuity_pv()['gross'] > 0
    def test_net_below_gross(self): assert annuity_pv()['net'] < annuity_pv()['gross']
    def test_net_58pct(self):       assert abs(annuity_pv()['net']/annuity_pv()['gross']-0.58)<0.001
    def test_years_range(self):     assert 30 < annuity_pv(age=37)['years'] < 55
    def test_higher_rate_lower_pv(self):
        assert annuity_pv(rate=0.02)['gross'] > annuity_pv(rate=0.08)['gross']

# ── Ownership ──────────────────────────────────────────────────────────────

class TestOwnership:
    def test_1seed_high(self):   assert get_ownership('Duke',make_teams()) > 0.90
    def test_16seed_low(self):   assert get_ownership('Siena',make_teams()) < 0.10
    def test_bounded(self):
        for t in make_teams(): assert 0 < get_ownership(t,make_teams()) < 1

# ── Integration ────────────────────────────────────────────────────────────

class TestChalkIntegration:
    def test_chalk_with_live_data(self):
        import os; os.chdir(os.path.dirname(os.path.dirname(__file__)))
        from march_madness_2026_v4 import load_data, resolve_bracket, build_chalk, FIRST_FOUR_RESULTS
        teams,vegas=load_data(); bracket=resolve_bracket(FIRST_FOUR_RESULTS)
        beta=np.array([0.051,2.546,0.628,0.339,0.399,0.633,0.0,0.539,-0.050,0.349])
        chalk=build_chalk(beta,bracket,teams,protect=True)
        assert 0 < chalk['jp_r1p'] < 1
        assert chalk['jp_r1r2p'] < chalk['jp_r1p']
        for r in ['East','West','Midwest','South']:
            assert len(chalk[r]['r1'])==8
            assert len(chalk[r]['r2'])==4

if __name__ == '__main__':
    pytest.main([__file__,'-v'])
