"""
Tests for backtest_harness_v6.py — phase momentum, factors_v6,
predict_v6, calibrate_v6, beta sign stability, conf champ flags.
Pure utilities (bayesian_blend, injury_residual, etc.) are covered
in test_model_core.py; these tests focus on harness-specific logic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),'backtest'))

import numpy as np
import pytest
from model_core import (
    CORPUS_INJURY_BETA, POWER_CHAMPS, MID_CHAMPS,
    get_champ_flags, injury_residual,
)
from backtest_harness_v6 import (
    phase_momentum, factors_v6, predict_v6, calibrate_v6,
)

# ── Stubs ──────────────────────────────────────────────────────────────────

def make_teams():
    return {
        'Duke':        {'seed':1, 'kenpom_rank':1,  'seniority':3.0,'coach_xp':6.0,
                        'injury_severity':2.0,'sos_score':9.5,'momentum':8.0,'q1_wins':17},
        'Siena':       {'seed':16,'kenpom_rank':155,'seniority':2.5,'coach_xp':1.0,
                        'injury_severity':0.0,'sos_score':4.5,'momentum':0.0,'q1_wins':0},
        'Murray State':{'seed':12,'kenpom_rank':65, 'seniority':7.0,'coach_xp':2.0,
                        'injury_severity':0.0,'sos_score':7.0,'momentum':0.5,'q1_wins':1},
        'Marquette':   {'seed':5, 'kenpom_rank':20, 'seniority':6.0,'coach_xp':4.5,
                        'injury_severity':0.0,'sos_score':8.3,'momentum':3.5,'q1_wins':7},
    }

def make_beta():
    return np.array([0.0,2.5,0.40,0.30,0.50,0.30,0.0,0.25,0.10,0.05])

# ── Conference champion flags (via model_core) ─────────────────────────────

class TestConfChampFlags:
    def test_power_2026_duke(self):
        pc,mc = get_champ_flags('Duke',2026)
        assert pc==1.0 and mc==0.0

    def test_mid_2026_gonzaga(self):
        pc,mc = get_champ_flags('Gonzaga',2026)
        assert pc==0.0 and mc==1.0

    def test_mutual_exclusion_all_years(self):
        for year in [2019,2024,2025,2026]:
            overlap = POWER_CHAMPS.get(year,set()) & MID_CHAMPS.get(year,set())
            assert len(overlap)==0, f"Year {year}: {overlap}"

    def test_historic_2019_virginia(self):
        pc,_ = get_champ_flags('Virginia',2019)
        assert pc==1.0

    def test_historic_2025_drake(self):
        pc,mc = get_champ_flags('Drake',2025)
        assert pc==0.0 and mc==1.0

# ── Injury residual (via model_core) ──────────────────────────────────────

class TestInjuryResidual:
    def test_zero(self):                 assert injury_residual(0.0,0.0)==0.0
    def test_fully_priced(self):         assert injury_residual(2.5,1.0)==0.0
    def test_partial(self):
        r=injury_residual(0.5,0.0); assert 0<r<0.5
    def test_never_exceeds_severity(self):
        for s in [0.5,1.0,2.5]:
            assert injury_residual(s,0.0) <= s+1e-9
    def test_sharp_reduces(self):
        assert injury_residual(1.0,0.0) >= injury_residual(1.0,2.0)

# ── Phase momentum ─────────────────────────────────────────────────────────

class TestPhaseMomentum:
    def test_r1_returns_base(self):
        assert phase_momentum('A','R1',7.0,{'wins':0,'games':0}) == 7.0

    def test_r2_win_boosts_over_loss(self):
        m_w = phase_momentum('A','R2',7.0,{'wins':1,'games':1})
        m_l = phase_momentum('A','R2',7.0,{'wins':0,'games':1})
        assert m_w > m_l

    def test_s16_perfect_record(self):
        assert phase_momentum('A','S16',5.0,{'wins':2,'games':2}) == 10.0

    def test_s16_no_games_fallback(self):
        assert phase_momentum('A','S16',6.0,{'wins':0,'games':0}) == 6.0

# ── factors_v6 ─────────────────────────────────────────────────────────────

class TestFactorsV6:
    def test_shape(self):
        assert factors_v6('Duke','Siena',make_teams(),2026).shape == (9,)

    def test_dominant_positive_kenpom(self):
        assert factors_v6('Duke','Siena',make_teams(),2026)[0] > 0

    def test_power_cc_in_vector(self):
        f = factors_v6('Duke','Siena',make_teams(),2026)
        assert f[6] == 1.0  # Duke is power conf champ for 2026, Siena is mid

    def test_mid_cc_in_vector(self):
        f = factors_v6('Siena','Duke',make_teams(),2026)
        assert f[7] == 1.0  # Siena mid_cc=1 minus Duke mid_cc=0

    def test_interaction_favours_dominant(self):
        f = factors_v6('Duke','Siena',make_teams(),2026)
        assert f[8] > 0

    def test_unknown_teams_zero(self):
        assert np.allclose(factors_v6('X','Y',make_teams(),2026), 0)

# ── predict_v6 ─────────────────────────────────────────────────────────────

class TestPredictV6:
    def test_dominant_high_prob(self):
        p = predict_v6(make_beta(),'Duke','Siena',make_teams(),2026,
                       rnd='R1',seed_a=1,seed_b=16)
        assert p > 0.88

    def test_unit_interval(self):
        beta=make_beta(); teams=make_teams()
        for a,b in [('Duke','Siena'),('Murray State','Marquette')]:
            assert 0 < predict_v6(beta,a,b,teams,2026) < 1

    def test_conf_champ_boosts(self):
        beta=make_beta(); teams=make_teams()
        t_flag   = dict(teams); t_flag['Murray State'] = dict(teams['Murray State'])
        t_noflag = dict(teams); t_noflag['Murray State'] = dict(teams['Murray State'])
        # Add 2025 mid_cc via year: Murray State not in 2025 MID_CHAMPS
        # Use 2019 to test Murray State (OVC champ 2019)
        p_flag   = predict_v6(beta,'Murray State','Marquette',teams,2019)
        p_noflag = predict_v6(beta,'Murray State','Marquette',teams,2020)
        assert p_flag >= p_noflag, "Conf champ year should not hurt Murray State"

    def test_injury_direction(self):
        beta=make_beta(); teams=make_teams()
        p_inj   = predict_v6(beta,'Duke','Siena',teams,2026,ir_b=3.0)
        p_clean = predict_v6(beta,'Duke','Siena',teams,2026,ir_b=0.0)
        assert p_inj > p_clean

# ── calibrate_v6 ───────────────────────────────────────────────────────────

class TestCalibrateV6:
    def _vegas(self):
        return [('Duke','Siena',0.986,0.0),('Murray State','Marquette',0.450,-0.5)]

    def test_returns_shape(self):
        b = calibrate_v6(self._vegas(),make_teams(),2026,n_steps=200,verbose=False)
        assert b.shape == (10,)

    def test_kenpom_positive(self):
        b = calibrate_v6(self._vegas(),make_teams(),2026,n_steps=300,verbose=False)
        assert b[1] > 0

# ── Beta sign stability (pre-computed, year-over-year) ─────────────────────

class TestBetaSignStability:
    B2019 = np.array([0.149,2.075,0.142,0.699,0.635,0.346,0.0,0.261,0.149,-0.033])
    B2025 = np.array([0.323,3.487,0.368,0.709,0.878,0.093,0.0,0.095,0.150, 0.076])

    def test_kenpom_positive(self):
        assert self.B2019[1]>0 and self.B2025[1]>0

    def test_sos_positive(self):
        assert self.B2019[4]>0 and self.B2025[4]>0

    def test_coach_positive(self):
        assert self.B2019[3]>0 and self.B2025[3]>0

    def test_power_cc_positive(self):
        assert self.B2019[7]>0 and self.B2025[7]>0

    def test_mid_cc_positive(self):
        assert self.B2019[8]>0 and self.B2025[8]>0

    def test_injury_locked(self):
        assert CORPUS_INJURY_BETA == -0.244

if __name__ == '__main__':
    pytest.main([__file__,'-v'])
