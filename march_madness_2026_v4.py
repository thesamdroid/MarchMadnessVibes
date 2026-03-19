"""
March Madness 2026 — Bracket Optimizer v4
==========================================
Model:
    9-factor logistic regression calibrated on Vegas R1 lines.
    Shared utilities (Bayesian prior, sigmoid temperature, chaos epsilon,
    injury residual, conf-champ flags) live in model_core.py.

    Key features:
      • Log-compressed KenPom, CoachXP, SOS, Q1 momentum
      • Bayesian seed-matchup prior blend (R1 only)
      • Power vs mid-major conference champion split
      • Round-specific sigmoid temperature (T: 1.0 R1 → 0.70 FF)
      • Student-t chaos epsilon df=4 (heavy tails)
      • KenPom × Q1_Momentum interaction term
      • Injury residual beta LOCKED at -0.244 (empirical corpus)
      • Portal-corrected seniority (in CSV)

Bracket strategy:
    R1+R2  : chalk picks (maximise P(perfect R1) and P(perfect R1+R2))
    S16+   : game-by-game model WP with 70% upset threshold in R1
             (any R1 game where favourite raw WP < 70% → pick the upset,
              except 1v16 which is always protected)
    FF+    : game-by-game model WP, no threshold

    Backwards compatibility enforced: every FF pick = that region's E8 winner,
    champion = one of the four regional champions.

Usage:
    python march_madness_2026_v4.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit
import warnings
warnings.filterwarnings("ignore")

from model_core import (
    CORPUS_INJURY_BETA, ROUND_TEMP, SEED_PRIORS, SEED_ALPHA,
    POWER_CHAMPS, MID_CHAMPS,
    log_norm, log_rank,
    get_seed_prior, bayesian_blend,
    tempered_sig, chaos_eps,
    vegas_coverage, injury_residual,
    get_champ_flags,
)

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

FIRST_FOUR_RESULTS = {
    'UMBC_vs_Howard':    'Howard',   # FINAL: Howard 86, UMBC 83
    'NC_State_vs_Texas': 'Texas',    # FINAL: Texas 68, NC State 66 (Tramon Mark buzzer)
    'Lehigh_vs_PV':      'Pr. View', # FINAL: Prairie View A&M 67, Lehigh 55
    'SMU_vs_MiamiOH':    'Miami OH',  # FINAL: Miami OH 63, SMU 55
}

PRIZE_R1R2 = 19_800_000
PRIZE_R1   =  1_000_000
PROTECT_1V16 = True
N_ITER     = 100_000
GD_STEPS   =  80_000
AGE        = 37
DISC_RATE  = 0.04

# =============================================================================
# BRACKET STRUCTURE
# =============================================================================

BRACKET = {
    'East': [
        ('Duke','Siena'), ('Ohio State','TCU'), ("St. John's",'N. Iowa'),
        ('Kansas','Cal Baptist'), ('Louisville','S. Florida'),
        ('Mich. State','ND State'), ('UCLA','UCF'), ('UConn','Furman'),
    ],
    'West': [
        ('Arizona','LIU'), ('Villanova','Utah State'), ('Wisconsin','High Point'),
        ('Arkansas','Hawaii'), ('BYU',None), ('Gonzaga','Kennesaw St.'),
        ('Miami FL','Missouri'), ('Purdue','Queens'),
    ],
    'Midwest': [
        ('Michigan',None), ('Georgia','Saint Louis'), ('Texas Tech','Akron'),
        ('Alabama','Hofstra'), ('Tennessee',None), ('Virginia','Wright State'),
        ('Kentucky','Santa Clara'), ('Iowa State','Tenn. State'),
    ],
    'South': [
        ('Florida',None), ('Iowa','Clemson'), ('Vanderbilt','McNeese'),
        ('Nebraska','Troy'), ('N. Carolina','VCU'), ('Illinois','Penn'),
        ("Saint Mary's",'Texas A&M'), ('Houston','Idaho'),
    ],
}

FIRST_FOUR = [
    ('UMBC','Howard','Midwest',0),
    ('NC State','Texas','West',4),
    ('Lehigh','Pr. View','South',0),
    ('SMU','Miami OH','Midwest',4),
]

FF_PAIRS   = [('East','South'),('West','Midwest')]
REGIONS    = ['East','West','Midwest','South']
GAME_1V16  = 0
ROUND_PTS  = [1,2,4,8,16,32]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def rg(a, b):
    if a is None: return b, b
    if b is None: return a, a
    return a, b

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    _here = Path(__file__).parent
    teams_df = pd.read_csv(_here / 'teams_2026.csv')
    vegas_df = pd.read_csv(_here / 'vegas_lines_2026.csv')
    teams = teams_df.set_index('team').to_dict('index')
    vegas = [(r['team_a'],r['team_b'],r['p_a_wins'],r['sharp_move_pts'])
             for _,r in vegas_df.iterrows()]
    return teams, vegas

def resolve_bracket(ff_results):
    slot = {
        0: ff_results.get('UMBC_vs_Howard'),
        1: ff_results.get('NC_State_vs_Texas'),
        2: ff_results.get('Lehigh_vs_PV'),
        3: ff_results.get('SMU_vs_MiamiOH'),
    }
    bk = {r: list(g) for r,g in BRACKET.items()}
    for fi, (ta,tb,region,gi) in enumerate(FIRST_FOUR):
        w = slot.get(fi)
        if w:
            a,b = bk[region][gi]
            bk[region][gi] = (w,b) if a is None else (a,w)
    return bk

# =============================================================================
# v6 FACTOR VECTOR + PREDICTION
# =============================================================================

def factors(a, b, teams, mp_a=None, mp_b=None, mt_a=None, mt_b=None):
    ta, tb = teams.get(a,{}), teams.get(b,{})
    if not ta or not tb: return np.zeros(9)

    ra, rb = ta.get('kenpom_rank',100), tb.get('kenpom_rank',100)
    kenpom = log_rank(rb) - log_rank(ra)

    mp_a_ = mp_a if mp_a is not None else ta.get('momentum',5.0)
    mp_b_ = mp_b if mp_b is not None else tb.get('momentum',5.0)
    mt_a_ = mt_a if mt_a is not None else 0.0
    mt_b_ = mt_b if mt_b is not None else 0.0

    # KenPom × Momentum interaction: "dominant AND hot"
    def kp_mom(team, t_dict, mom):
        kp_str = log_norm(max(0, 200 - t_dict.get('kenpom_rank',100)), scale=200)
        return kp_str * log_norm(mom, scale=10)

    interact = kp_mom(a, ta, mp_a_) - kp_mom(b, tb, mp_b_)

    return np.array([
        kenpom,
        (ta['seniority']        - tb['seniority'])        / 10.0,
        log_norm(ta['coach_xp'])- log_norm(tb['coach_xp']),
        log_norm(ta['sos_score'])- log_norm(tb['sos_score']),
        log_norm(mp_a_)         - log_norm(mp_b_),
        (mt_a_                  - mt_b_)                  / 10.0,
        float(ta.get('power_cc',0)) - float(tb.get('power_cc',0)),
        float(ta.get('mid_cc',0))   - float(tb.get('mid_cc',0)),
        interact,
    ])

def predict(beta, a, b, teams, rnd='R1',
            mp_a=None, mp_b=None, mt_a=None, mt_b=None,
            ir_a=0.0, ir_b=0.0, seed_a=None, seed_b=None):
    if not a or not b or a==b: return 0.99
    f  = factors(a, b, teams, mp_a, mp_b, mt_a, mt_b)
    logit = beta[0] + np.dot(beta[1:], f)
    # Locked injury adjustment
    # ir_a - ir_b: opponent injury helps team A (positive delta → higher logit)
    logit += CORPUS_INJURY_BETA * 3.0 * ((ir_a - ir_b) / 3.0)
    p = tempered_sig(logit, rnd)
    # Bayesian prior blend (R1 only)
    if rnd == 'R1' and seed_a is not None:
        p = bayesian_blend(p, seed_a, seed_b)
    return p

def get_k(a, b, teams):
    q = teams.get(a,{}).get('q1_wins',0) + teams.get(b,{}).get('q1_wins',0)
    return float(np.clip(8 + q*1.3, 6, 40))

# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate(vegas, teams, n_steps=GD_STEPS, verbose=True):
    beta = np.array([0.0, 2.5, 0.40, 0.30, 0.50, 0.30, 0.0, 0.25, 0.10, 0.05])
    for step in range(n_steps):
        lr = 0.05 if step<35000 else 0.02 if step<60000 else 0.005
        grad = np.zeros(10)
        for a, b, vp, sm in vegas:
            ta = teams.get(a,{}); tb = teams.get(b,{})
            sa, sb = ta.get('seed'), tb.get('seed')
            f = factors(a, b, teams, mt_a=0.0, mt_b=0.0)
            logit = beta[0] + np.dot(beta[1:], f)
            p_raw = tempered_sig(logit, 'R1')
            p_blended = bayesian_blend(p_raw, sa, sb)
            e = p_blended - vp
            prior, alpha = (SEED_PRIORS.get((min(sa,sb),max(sa,sb)), None),
                            SEED_ALPHA.get((min(sa,sb),max(sa,sb)), 1.0)) \
                           if sa and sb else (None, 1.0)
            alpha = alpha if prior else 1.0
            dsig  = p_raw * (1.0-p_raw) / ROUND_TEMP.get('R1',1.0)
            grad[0]  += e * alpha * dsig
            grad[1:] += e * alpha * dsig * f
        beta -= lr * grad / max(len(vegas),1)
        if verbose and (step+1) % 20000 == 0:
            loss = sum(-(vp*np.log(np.clip(bayesian_blend(
                tempered_sig(beta[0]+np.dot(beta[1:],
                factors(a,b,teams,mt_a=0,mt_b=0)),'R1'),
                teams.get(a,{}).get('seed'),teams.get(b,{}).get('seed')),
                1e-9,1-1e-9))+(1-vp)*np.log(np.clip(1-bayesian_blend(
                tempered_sig(beta[0]+np.dot(beta[1:],
                factors(a,b,teams,mt_a=0,mt_b=0)),'R1'),
                teams.get(a,{}).get('seed'),teams.get(b,{}).get('seed')),
                1e-9,1-1e-9))) for a,b,vp,sm in vegas)/len(vegas)
            print(f"  Step {step+1:>6,} | Loss: {loss:.5f}")
    labels = ['bias','KenPom_log','Seniority','CoachXP_log','SOS_log',
              'Mom_pre_log','Mom_tourney','PowerCC','MidCC','KpxMom']
    print(f"\nCalibrated | β: " +
          " ".join(f"{l}={b:+.3f}" for l,b in zip(labels,beta)))
    print(f"  InjResid={CORPUS_INJURY_BETA:+.3f} [LOCKED]")
    return beta

# =============================================================================
# ANNUITY
# =============================================================================

def annuity_pv(payment=1_000_000, age=AGE, rate=DISC_RATE):
    def q(a):
        if a<45: return 0.00175+a*0.000020
        if a<55: return 0.00300+a*0.000050
        if a<65: return 0.00600+a*0.000100
        if a<75: return 0.01500+(a-65)*0.002
        if a<85: return 0.03800+(a-75)*0.006
        return min(0.35, 0.10+(a-85)*0.015)
    pv,s,yrs = 0.0,1.0,0.0
    for t in range(80):
        pv += s*payment/((1+rate)**t); yrs += s; s*=(1-q(age+t))
        if s<0.001: break
    return {'gross':pv,'net':pv*0.58,'years':yrs}

# =============================================================================
# CHALK PICKS
# =============================================================================

def build_chalk(beta, bracket, teams, protect=PROTECT_1V16):
    chalk = {}
    jp_r1=1.0; jp_r1p=1.0; jp_r1r2=1.0; jp_r1r2p=1.0
    prot_set = {(r,GAME_1V16) for r in REGIONS} if protect else set()

    for region in REGIONS:
        r1_picks,r1_wps,r1_prot = [],[],[]
        for gi,(a,b) in enumerate(bracket[region]):
            ta,tb = rg(a,b)
            td = teams.get(ta,{}); sd = teams.get(tb,{})
            sa = td.get('seed'); sb = sd.get('seed')
            wp = predict(beta,ta,tb,teams,rnd='R1',seed_a=sa,seed_b=sb)
            pick = ta if wp>=0.5 else tb; win_p = max(wp,1-wp)
            is_1v16 = (gi==GAME_1V16)
            r1_picks.append(pick); r1_wps.append(win_p); r1_prot.append(is_1v16)
            jp_r1 *= win_p; jp_r1r2 *= win_p
            if not (protect and is_1v16):
                jp_r1p *= win_p; jp_r1r2p *= win_p

        r2_picks,r2_wps = [],[]
        for i in range(0,8,2):
            wp = predict(beta,r1_picks[i],r1_picks[i+1],teams,rnd='R2')
            pick = r1_picks[i] if wp>=0.5 else r1_picks[i+1]
            win_p = max(wp,1-wp)
            r2_picks.append(pick); r2_wps.append(win_p)
            jp_r1r2 *= win_p; jp_r1r2p *= win_p

        chalk[region] = {'r1':r1_picks,'r1_wp':r1_wps,'r1_prot':r1_prot,
                         'r2':r2_picks,'r2_wp':r2_wps}

    chalk.update({'jp_r1':jp_r1,'jp_r1p':jp_r1p,
                  'jp_r1r2':jp_r1r2,'jp_r1r2p':jp_r1r2p})
    return chalk

# =============================================================================
# MONTE CARLO — with Student-t chaos + tempered sigmoid
# =============================================================================

def simulate(beta, bracket, chalk, teams, n=N_ITER,
             protect=PROTECT_1V16, verbose=True):
    rng = np.random.default_rng(seed=42)

    chalk_r2 = sum([chalk[r]['r2'] for r in REGIONS], [])
    prot_set  = {(r,GAME_1V16) for r in REGIONS} if protect else set()

    ff_chalk = {}
    for fi,(ta,tb,region,gi) in enumerate(FIRST_FOUR):
        a,b = rg(*bracket[region][gi])
        wp = predict(beta,a,b,teams,rnd='R1',
                     seed_a=teams.get(a,{}).get('seed'),
                     seed_b=teams.get(b,{}).get('seed'))
        ff_chalk[fi] = a if wp>=0.5 else b

    counts        = {t:[0]*6 for t in teams}
    region_counts = {r:{t:[0]*4 for t in teams} for r in REGIONS}
    ff_counts     = {t:0 for t in teams}
    ch_counts     = {t:0 for t in teams}
    perf_r1 = perf_r12 = 0

    def sim_g(a, b, eps, rnd):
        ta,tb = rg(a,b)
        if ta==tb: return ta
        sa = teams.get(ta,{}).get('seed')
        sb = teams.get(tb,{}).get('seed')
        p  = predict(beta,ta,tb,teams,rnd=rnd,seed_a=sa,seed_b=sb)
        k  = get_k(ta,tb,teams)
        # Use logit space for eps perturbation, then retransform
        logit_adj = np.log(p/(1-p)) + eps
        p_adj  = float(expit(logit_adj))
        std    = np.sqrt(p_adj*(1-p_adj)/k)
        wp     = float(np.clip(p_adj + std*rng.standard_normal(), 0.01, 0.99))
        return ta if rng.random()<wp else tb

    for it in range(n):
        # Student-t chaos epsilon (heavier tails than Normal)
        eps = chaos_eps(rng)

        ff_res,ff_ok = [],True
        for fi,(ta,tb,region,gi) in enumerate(FIRST_FOUR):
            a,b = rg(*bracket[region][gi])
            w = sim_g(a,b,0.0,'R1')
            ff_res.append(w)
            if w != ff_chalk.get(fi,w): ff_ok = False

        bk = {r:list(bracket[r]) for r in REGIONS}
        for fi,(_,_,region,gi) in enumerate(FIRST_FOUR):
            a,b = bk[region][gi]
            if a is None:   bk[region][gi] = (ff_res[fi],b)
            elif b is None: bk[region][gi] = (a,ff_res[fi])

        all_r1,all_r2,e8 = [],[],{}
        for region in REGIONS:
            r1w = []
            for gi,(a,b) in enumerate(bk[region]):
                w = sim_g(a,b,eps,'R1')
                r1w.append(w)
                if w in counts: counts[w][0]+=1
                if w in region_counts[region]: region_counts[region][w][0]+=1

            r2w = [sim_g(r1w[i],r1w[i+1],eps,'R2') for i in range(0,8,2)]
            for w in r2w:
                if w in counts: counts[w][1]+=1
                if w in region_counts[region]: region_counts[region][w][1]+=1

            s16 = [sim_g(r2w[0],r2w[1],eps,'S16'),
                   sim_g(r2w[2],r2w[3],eps,'S16')]
            for w in s16:
                if w in counts: counts[w][2]+=1
                if w in region_counts[region]: region_counts[region][w][2]+=1

            e8w = sim_g(s16[0],s16[1],eps,'E8')
            if e8w in counts: counts[e8w][3]+=1
            if e8w in region_counts[region]: region_counts[region][e8w][3]+=1
            e8[region]=e8w; all_r1.extend(r1w); all_r2.extend(r2w)

        ff1 = sim_g(e8['East'], e8['South'],  0.0,'FF')
        ff2 = sim_g(e8['West'], e8['Midwest'],0.0,'FF')
        for w in [ff1,ff2]:
            ff_counts[w]=ff_counts.get(w,0)+1
            if w in counts: counts[w][4]+=1
        ch = sim_g(ff1,ff2,0.0,'Champ')
        ch_counts[ch]=ch_counts.get(ch,0)+1
        if ch in counts: counts[ch][5]+=1

        # Perfect round check
        flat=0; r1_ok=ff_ok
        for region in REGIONS:
            for gi in range(8):
                result   = all_r1[flat]
                expected = chalk[region]['r1'][gi]
                if not ((region,gi) in prot_set) and result!=expected:
                    r1_ok=False
                flat+=1
        if r1_ok:
            perf_r1+=1
            if all(a==b for a,b in zip(all_r2,chalk_r2)): perf_r12+=1

        if verbose and (it+1)%10000==0:
            print(f"  [{it+1:>7,}] P(R1): {perf_r1/(it+1)*100:.4f}%  "
                  f"P(R1+R2): {perf_r12/(it+1)*100:.5f}%")

    return {'counts':counts,'region_counts':region_counts,
            'ff_counts':ff_counts,'ch_counts':ch_counts,
            'perf_r1':perf_r1,'perf_r12':perf_r12,'n':n}

# =============================================================================
# FIVE-BRACKET OPTIMIZER (unchanged logic, uses updated predict())
# =============================================================================

UPSET_THRESHOLD = 0.70   # R1 only: pick upset if favourite raw model WP < this


def play_game(beta, a, b, teams, rnd, game_index=None):
    """
    Pick winner of a single matchup.

    R1 logic (game_index provided):
      - index 0 is the 1v16 protected game → always chalk
      - otherwise: if favourite's RAW model WP < UPSET_THRESHOLD, pick the upset
        Raw WP used (pre-Bayesian-blend) so the model's genuine team-quality
        opinions drive upset picks rather than being smoothed toward seed history.

    R2+ logic: straight model WP, no threshold.
    """
    if not a or not b:
        return (a or b), None, 1.0
    ta = teams.get(a, {}); tb = teams.get(b, {})
    sa = ta.get('seed'); sb = tb.get('seed')

    if rnd == 'R1' and game_index is not None:
        fav = a if (sa or 8) <= (sb or 8) else b
        dog = b if fav == a else a
        sf  = min(sa or 8, sb or 8)

        # Raw logit (no Bayesian blend)
        f = factors(a, b, teams)
        raw_wp_a = float(expit(beta[0] + np.dot(beta[1:], f)))
        raw_fav  = raw_wp_a if (sa or 8) <= (sb or 8) else 1.0 - raw_wp_a

        protected = (game_index == 0)   # 1v16 always chalk
        if protected or raw_fav >= UPSET_THRESHOLD:
            return fav, dog, raw_fav
        else:
            return dog, fav, 1.0 - raw_fav   # upset

    # R2 and beyond: straight blended model WP
    wp = predict(beta, a, b, teams, rnd=rnd, seed_a=sa, seed_b=sb)
    return (a, b, wp) if wp >= 0.5 else (b, a, 1.0 - wp)


def optimize_region(region, sim, bracket, chalk, teams, n, beta=None):
    """
    Game-by-game bracket solver: play each matchup with per-game model WP.
    This surfaces genuine upsets the model believes in rather than always
    following cumulative survival probabilities that favour 1-seeds.

    Returns the same dict shape as before for backwards compatibility.
    """
    if beta is None:
        # Fallback: use chalk R2 winners as seeds for S16+
        r2 = chalk[region]['r2']
        return {'regional_champ': r2[0], 's16_top': r2[0], 's16_bot': r2[2],
                'pod_of_champ': 'top',
                'sim_pcts': {'s16_top': 0.5, 's16_bot': 0.5, 'e8': 0.5}}

    # R1 games: pass game_index so 1v16 is protected, others use upset threshold
    r1_winners = []
    r1_wps     = []
    r1_prots   = []
    for gi, (a, b) in enumerate(bracket[region]):
        a2, b2 = (a or b), (b or a)
        w, l, wp = play_game(beta, a2, b2, teams, 'R1', game_index=gi)
        r1_winners.append(w)
        r1_wps.append(wp)
        r1_prots.append(gi == 0)

    # R2: game-by-game, no threshold
    r2w = []
    r2_wps = []
    for i in range(0, 8, 2):
        w, l, wp = play_game(beta, r1_winners[i], r1_winners[i+1], teams, 'R2')
        r2w.append(w); r2_wps.append(wp)

    # S16: pod0 vs pod1, pod2 vs pod3
    s16_w0, s16_l0, s16_wp0 = play_game(beta, r2w[0], r2w[1], teams, 'S16')
    s16_w1, s16_l1, s16_wp1 = play_game(beta, r2w[2], r2w[3], teams, 'S16')

    # E8
    e8_w, e8_l, e8_wp = play_game(beta, s16_w0, s16_w1, teams, 'E8')

    # Determine which S16 pod the regional champ came from
    pod = 'top' if e8_w == s16_w0 else 'bot'

    rc = sim['region_counts'][region]
    def sp(t, rnd): return rc.get(t, [0]*4)[rnd] / n

    return {'regional_champ': e8_w,
            's16_top': s16_w0,
            's16_bot': s16_w1,
            'pod_of_champ': pod,
            'r1_winners': r1_winners,
            'r1_wps':     r1_wps,
            'r1_prots':   r1_prots,
            'r2_winners': r2w,
            'r2_wps':     r2_wps,
            'game_wps': {'s16_top': s16_wp0, 's16_bot': s16_wp1, 'e8': e8_wp},
            'sim_pcts': {'s16_top': sp(s16_w0, 2),
                         's16_bot': sp(s16_w1, 2),
                         'e8':      sp(e8_w, 3)}}


def optimize_ff(sim, region_opts, teams, n, beta=None):
    """
    Game-by-game Final Four: play each semifinal and the championship
    using per-game model WP. Backwards compatibility enforced:
    each region's FF pick = that region's regional_champ.
    """
    def ch_sim(team):
        return sim['ch_counts'].get(team, 0) / n

    regional_champs = {r: region_opts[r]['regional_champ'] for r in REGIONS}
    ff_picks = {r: regional_champs[r] for r in REGIONS}

    if beta is not None:
        # Play out FF semis game-by-game
        semi_winners = []
        for pair in FF_PAIRS:
            r1, r2 = pair
            t1, t2 = regional_champs[r1], regional_champs[r2]
            w, l, wp = play_game(beta, t1, t2, teams, 'FF')
            semi_winners.append((w, wp))
        champion, _, champ_wp = play_game(
            beta, semi_winners[0][0], semi_winners[1][0], teams, 'Champ')
    else:
        champion = max(regional_champs.values(), key=ch_sim)

    champ_ff_region = next(r for r, t in regional_champs.items() if t == champion)
    champ_ff_pair   = next(p for p in FF_PAIRS if champ_ff_region in p)

    return {'champion': champion,
            'ff_picks': ff_picks,
            'champ_pair': champ_ff_pair,
            'sim_pcts': {'champion': ch_sim(champion),
                         **{r: sim['ff_counts'].get(t, 0)/n
                            for r, t in ff_picks.items()}}}

# =============================================================================
# PRINTING
# =============================================================================
ROUND_PTS = [1,2,4,8,16,32]

def sep(t=''):
    print(f"\n{'='*68}")
    if t: print(f"  {t}")
    print(f"{'='*68}")

def print_summary(sim, chalk, ann, region_opts, ff_opt, bracket, protect, teams=None):
    n = sim['n']
    sep("EXPECTED VALUE ANALYSIS")
    pr1p = chalk['jp_r1p']; pr12p = chalk['jp_r1r2p']
    ev1  = pr12p * ann['gross']
    ev2  = (pr1p - pr12p) * PRIZE_R1
    ev3  = 250_000 / 100_000  # pool: $250K prize / ~100K entries
    print(f"\n  Annuity PV: ${ann['gross']:>14,.0f}  (net 42%: ${ann['net']:>12,.0f})")
    print(f"  P(perfect R1)    with protection: {pr1p*100:.5f}%")
    print(f"  P(perfect R1+R2) with protection: {pr12p*100:.6f}%")
    print(f"\n  {'Payout':<34} {'P(hit)':>10}  {'EV':>10}")
    print(f"  {'-'*58}")
    gross_m = f"{ann['gross']/1e6:.1f}"
    print(f"  {'Annuity ($'+gross_m+'M PV)':<34} {pr12p*100:>9.5f}%  ${ev1:>8,.0f}")
    print(f"  {'$1M lump (R1 perfect only)':<34} {(pr1p-pr12p)*100:>9.5f}%  ${ev2:>8,.0f}")
    print(f"  {'Pool ($250K / 100K entries)':<34} {'~0.001%':>10}  ${ev3:>8,.0f}")
    print(f"  Total EV: ${ev1+ev2+ev3:,.0f}")

    sep("CHALK PICKS — R1 + R2  (maximise P(perfect R1) and P(R1+R2))")
    print("  [P]=1v16 protected  | BYU/Tennessee/Florida R1 opponent TBD tonight")
    for region in REGIONS:
        c = chalk[region]
        print(f"\n  ── {region} ──────────────────────────────────")
        for gi,(pick,wp,prot) in enumerate(zip(c['r1'],c['r1_wp'],c['r1_prot'])):
            a,b = bracket[region][gi]
            opp = (b or 'FF-TBD') if a==pick else (a or 'FF-TBD')
            tag = ' [P]' if (protect and prot) else ''
            print(f"  R1: {pick:<24} vs {str(opp):<18} {wp*100:.1f}%{tag}")
        for i,(pick,wp) in enumerate(zip(c['r2'],c['r2_wp'])):
            print(f"  R2: {pick:<24} {wp*100:.1f}%")

    sep("BRACKET PICKS — S16 ONWARD  (per-game model WP)")
    print(f"  Champion: {ff_opt['champion']}  "
          f"(champ sim={sim['ch_counts'].get(ff_opt['champion'],0)/n*100:.1f}%)\n")
    for region in REGIONS:
        opt  = region_opts[region]
        champ = opt['regional_champ']
        top   = opt['s16_top']; bot = opt['s16_bot']
        wps   = opt.get('game_wps', {})
        sp    = opt.get('sim_pcts', {})
        print(f"  [{region[:1]}] {region:<10} Regional champ: {champ:<20} "
              f"E8 game={wps.get('e8',0)*100:.0f}%  (sim={sp.get('e8',0)*100:.0f}%)")
        for label, pick, wk, sk in [
            ('S16 top', top, 's16_top', 's16_top'),
            ('S16 bot', bot, 's16_bot', 's16_bot'),
        ]:
            flag = ' ← champ path' if pick == champ else ''
            print(f"       {label}: {pick:<20} game={wps.get(wk,0)*100:.0f}%  "
                  f"sim={sp.get(sk,0)*100:.0f}%{flag}")

    sep("FINAL FOUR BRACKET")
    ff = ff_opt['ff_picks']; ch = ff_opt['champion']
    for pair in FF_PAIRS:
        r1,r2 = pair
        t1,t2 = ff[r1],ff[r2]
        sp1 = sim['ff_counts'].get(t1,0)/n; sp2 = sim['ff_counts'].get(t2,0)/n
        c1  = ' ← champ' if t1==ch else ''; c2 = ' ← champ' if t2==ch else ''
        print(f"  {r1+'/'+r2:<22}: {t1:<22} ({sp1*100:.1f}% FF){c1}")
        print(f"  {'':22}  vs {t2:<22} ({sp2*100:.1f}% FF){c2}")
    chsp = sim['ch_counts'].get(ch,0)/n
    print(f"\n  CHAMPION: {ch}  ({chsp*100:.2f}% sim)")

    sep("GAME-BY-GAME PICKS — ALL ROUNDS")
    print(f"  [P]=1v16 protected  *** = upset pick  % = per-game model WP\n")
    upsets_count = 0
    for region in REGIONS:
        opt = region_opts[region]
        r1w = opt.get('r1_winners', chalk[region]['r1'])
        r1wp = opt.get('r1_wps',    [w*100 for w in chalk[region]['r1_wp']])
        r1pr = opt.get('r1_prots',  chalk[region]['r1_prot'])
        r2w  = opt.get('r2_winners', chalk[region]['r2'])
        r2wp = opt.get('r2_wps',    [w*100 for w in chalk[region]['r2_wp']])
        champ = opt['regional_champ']
        wps   = opt.get('game_wps', {})
        print(f"  ═══ {region.upper()} ═══")
        for gi, (pick, wp, prot) in enumerate(zip(r1w, r1wp, r1pr)):
            a, b = bracket[region][gi]
            # detect upset: winner is the higher seed number
            sa = teams.get(a, {}).get('seed', 0); sb = teams.get(b, {}).get('seed', 0)
            chalk_pick = a if sa <= sb else b
            is_upset = (pick != chalk_pick) and (gi != 0)
            tag = ' [P]' if prot else (' ***' if is_upset else '')
            if is_upset: upsets_count += 1
            print(f"  R1: {pick:<24} {wp*100:.0f}%{tag}" if isinstance(wp, float)
                  else f"  R1: {pick:<24} {wp:.0f}%{tag}")
        for pick, wp in zip(r2w, r2wp):
            print(f"  R2: {pick:<24} {wp*100:.0f}%" if isinstance(wp, float)
                  else f"  R2: {pick:<24} {wp:.0f}%")
        s16t, s16b = opt['s16_top'], opt['s16_bot']
        print(f" S16: {s16t:<24} {wps.get('s16_top',0)*100:.0f}%" +
              (' ← champ path' if s16t == champ else ''))
        print(f" S16: {s16b:<24} {wps.get('s16_bot',0)*100:.0f}%" +
              (' ← champ path' if s16b == champ else ''))
        print(f"  E8: {champ:<24} {wps.get('e8',0)*100:.0f}%  ← Regional Champion")
        print()
    ff = ff_opt['ff_picks']; ch = ff_opt['champion']
    for region, team in ff.items():
        sp_ff = sim['ff_counts'].get(team, 0)/n
        flag = ' ← champion' if team == ch else ''
        print(f"  FF {region:<10}: {team:<22} {sp_ff*100:.0f}% sim{flag}")
    print(f"\n  CHAMPION: {ch}  ({sim['ch_counts'].get(ch,0)/n*100:.1f}% sim)")
    if upsets_count:
        print(f"\n  Upsets predicted in R1: {upsets_count}")

    ff_done = sum(1 for v in FIRST_FOUR_RESULTS.values() if v)
    if ff_done < 4:
        sep("SUBMISSION TIMING")
        remaining = [k for k,v in {'Lehigh_vs_PV':None,'SMU_vs_MiamiOH':None}.items()
                     if not FIRST_FOUR_RESULTS.get(k)]
        print(f"\n  ⚠  {4-ff_done} First Four games still tonight — DO NOT SUBMIT YET")
        print(f"  Confirmed: Howard ✓  Texas ✓")
        print(f"  Tonight:   Lehigh vs Pr. View (6:40 ET)  |  SMU vs Miami OH (9:15 ET)")
        print(f"\n  Add winners to FIRST_FOUR_RESULTS dict, rerun, then submit.")
        print(f"  Hard deadline: Thursday noon ET")

# =============================================================================
# MAIN
# =============================================================================

def main():
    sep("March Madness 2026 — Bracket Optimizer v4")
    print("  v6 features: log transforms · Bayes priors · power/mid CC · ")
    print("  sigmoid temp · t-chaos · interaction · locked injury β")

    print("\nLoading data...")
    teams, vegas = load_data()
    bracket = resolve_bracket(FIRST_FOUR_RESULTS)
    ff_done = sum(1 for v in FIRST_FOUR_RESULTS.values() if v)
    print(f"  {len(teams)} teams  |  {len(vegas)} Vegas lines  |  {ff_done}/4 First Four resolved")

    print(f"\nCalibrating ({GD_STEPS:,} steps)...")
    beta = calibrate(vegas, teams, GD_STEPS, verbose=True)

    print(f"\nAnnuity PV (age {AGE}, {DISC_RATE*100:.0f}% discount)...")
    ann = annuity_pv()
    print(f"  Gross: ${ann['gross']:>14,.0f}  Net: ${ann['net']:>12,.0f}")

    print(f"\nBuilding chalk picks (1v16 protect={PROTECT_1V16})...")
    chalk = build_chalk(beta, bracket, teams, PROTECT_1V16)
    print(f"  P(perfect R1):   {chalk['jp_r1p']*100:.5f}%")
    print(f"  P(perfect R1+R2):{chalk['jp_r1r2p']*100:.6f}%")

    print(f"\nRunning {N_ITER:,} Monte Carlo iterations (Student-t chaos)...")
    sim = simulate(beta, bracket, chalk, teams, N_ITER, PROTECT_1V16, verbose=True)
    n   = sim['n']
    print(f"  P(perfect R1)    simulated: {sim['perf_r1']/n*100:.4f}%")

    print("\nBuilding game-by-game bracket picks...")
    region_opts = {r: optimize_region(r,sim,bracket,chalk,teams,n,beta) for r in REGIONS}
    for r,opt in region_opts.items():
        wp = opt.get('game_wps',{}).get('e8', 0)
        print(f"  {r}: {opt['regional_champ']}  (E8 game WP {wp*100:.0f}%)")

    ff_opt = optimize_ff(sim, region_opts, teams, n, beta)
    print(f"  Champion: {ff_opt['champion']}")

    print_summary(sim, chalk, ann, region_opts, ff_opt, bracket, PROTECT_1V16, teams)

if __name__ == '__main__':
    main()
