"""
March Madness 2026 — Bracket Optimizer v4
==========================================
Incorporates all validated backtesting improvements (v6 harness features):

  TIER 1:
    • Bayesian seed-matchup prior blend on R1 predictions
    • Power vs mid-major conference champion split (two separate features)

  TIER 2:
    • Round-specific sigmoid temperature (T: 1.0→0.70 across rounds)
    • Student-t chaos epsilon df=4 (heavier tails for chaos year simulation)
    • KenPom × Q1_Momentum interaction term

  FACTOR UPDATES:
    • KenPom delta: log-compressed (already v3)
    • CoachXP: linear → log-compressed
    • SOS: linear → log-compressed
    • Q1 momentum: linear → log-compressed
    • Injury residual beta: LOCKED at -0.244 (empirical corpus, not calibrated)
    • Seniority: portal-corrected (in CSV, used linearly)

  FIVE-BRACKET ARCHITECTURE UNCHANGED:
    4 independent regional brackets + 1 Final Four bracket
    Champion → backwards-compatible path selection

Usage:
    python march_madness_2026_v4.py

    Update FIRST_FOUR_RESULTS after Wednesday night games,
    then rerun before Thursday noon ET deadline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from model_core import (
    CORPUS_INJURY_BETA, ROUND_TEMP, SEED_PRIORS, SEED_ALPHA,
    POWER_CHAMPS, MID_CHAMPS,
    log_norm, log_rank,
    get_seed_prior, bayesian_blend,
    tempered_sig, chaos_eps,
    vegas_coverage, injury_residual,
    get_champ_flags, expit,
)

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

FIRST_FOUR_RESULTS = {
    'UMBC_vs_Howard':    'Howard',   # CONFIRMED: Howard 86, UMBC 83
    'NC_State_vs_Texas': 'Texas',    # CONFIRMED: Texas 68, NC State 66
    # Fill in Wednesday night:
    # 'Lehigh_vs_PV':   'Lehigh',   # 6:40 PM ET tonight
    # 'SMU_vs_MiamiOH': 'SMU',      # 9:15 PM ET tonight
}

PRIZE_R1R2 = 19_800_000
PRIZE_R1   =  1_000_000
PRIZE_POOL =    250_000
POOL_SIZE  =    100_000
PROTECT_1V16 = True
N_ITER     = 100_000
GD_STEPS   =  80_000
AGE        = 37
DISC_RATE  = 0.04
CONTRARIAN = 1.0

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

def get_ownership(team, teams):
    own = {1:.96,2:.87,3:.77,4:.67,5:.58,6:.52,7:.50,
           8:.51,9:.49,10:.40,11:.43,12:.34,13:.24,14:.17,15:.08,16:.02}
    brand = [0.0,0.04,0.09,0.15]
    d = teams.get(team,{})
    return float(np.clip(own.get(d.get('seed',8),0.5) + brand[d.get('brand_tier',0)], 0.01, 0.97))

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

def pool_ev_score(sim_p, ownership, pts, agg=CONTRARIAN):
    return sim_p * pts * (0.30/max(0.01,ownership))**agg

def optimize_region(region, sim, bracket, chalk, teams, n):
    rc = sim['region_counts'][region]
    r2 = chalk[region]['r2']

    def rev(team, rnd, pts):
        sp = rc.get(team,[0]*4)[rnd]/n
        if sp<0.001: return 0.0
        reach = rc.get(team,[0]*4)[max(0,rnd-1)]/n if rnd>0 else 1.0
        own   = max(0.01, get_ownership(team,teams)*reach)
        return pool_ev_score(sp,own,pts)

    def reachable(rnd, min_p=0.005):
        return [t for t,c in rc.items() if c[rnd]/n>=min_p]

    champ_candidates = reachable(3)
    reg_champ = max(champ_candidates, key=lambda t: rev(t,3,ROUND_PTS[3]),
                    default=r2[0])

    top_r1 = set(chalk[region]['r1'][:4])
    bot_r1 = set(chalk[region]['r1'][4:])

    if reg_champ in top_r1: pod='top'
    elif reg_champ in bot_r1: pod='bot'
    else:
        p_top = sum(predict(np.zeros(10),reg_champ,t,teams,rnd='S16')
                    for t in list(top_r1)[:2]) / 2
        p_bot = sum(predict(np.zeros(10),reg_champ,t,teams,rnd='S16')
                    for t in list(bot_r1)[:2]) / 2
        pod = 'top' if p_top>=p_bot else 'bot'

    if pod=='top':
        s16_top = reg_champ
        bot_c   = [t for t in reachable(2) if t in bot_r1] or list(bot_r1)
        s16_bot = max(bot_c, key=lambda t: rev(t,2,ROUND_PTS[2]), default=r2[2])
    else:
        s16_bot = reg_champ
        top_c   = [t for t in reachable(2) if t in top_r1] or list(top_r1)
        s16_top = max(top_c, key=lambda t: rev(t,2,ROUND_PTS[2]), default=r2[0])

    return {'regional_champ':reg_champ,'s16_top':s16_top,'s16_bot':s16_bot,
            'pod_of_champ':pod,
            'pool_evs':{'s16_top':rev(s16_top,2,ROUND_PTS[2]),
                        's16_bot':rev(s16_bot,2,ROUND_PTS[2]),
                        'e8':rev(reg_champ,3,ROUND_PTS[3])}}

def optimize_ff(sim, region_opts, teams, n):
    def ff_ev(team, pts):
        sp = sim['ch_counts'].get(team,0)/n if pts==ROUND_PTS[5] else \
             sim['ff_counts'].get(team,0)/n
        if sp<0.001: return 0.0
        reach = sim['ff_counts'].get(team,0)/n
        own   = max(0.01, get_ownership(team,teams)*reach)
        return pool_ev_score(sp,own,pts)

    champ_cands = [t for t,c in sim['ch_counts'].items() if c/n>=0.005]
    champion = max(champ_cands, key=lambda t: ff_ev(t,ROUND_PTS[5]),
                   default=list(region_opts.values())[0]['regional_champ'])

    champ_ff_pair=None; champ_ff_region=None
    for pair in FF_PAIRS:
        for region in pair:
            region_teams=set()
            for a,b in BRACKET[region]:
                if a: region_teams.add(a)
                if b: region_teams.add(b)
            for _,_,ffr,_ in FIRST_FOUR:
                if ffr==region:
                    region_teams.update(['UMBC','Howard','NC State','Texas',
                                        'SMU','Miami OH','Lehigh','Pr. View'])
            if champion in region_teams and sim['region_counts'][region].get(champion,[0]*4)[3]/n>0.005:
                champ_ff_pair=pair; champ_ff_region=region; break
        if champ_ff_pair: break

    if not champ_ff_pair: champ_ff_pair=FF_PAIRS[0]; champ_ff_region=FF_PAIRS[0][0]

    ff_picks = {champ_ff_region: champion}
    other = [r for r in champ_ff_pair if r!=champ_ff_region][0]
    other_c = [t for t in champ_cands
               if sim['region_counts'][other].get(t,[0]*4)[3]/n>0.01] or champ_cands
    ff_picks[other] = max(other_c, key=lambda t: ff_ev(t,ROUND_PTS[4]),
                          default=region_opts[other]['regional_champ'])

    other_pair = [p for p in FF_PAIRS if p!=champ_ff_pair][0]
    for region in other_pair:
        c = [t for t in champ_cands
             if sim['region_counts'][region].get(t,[0]*4)[3]/n>0.01] or champ_cands
        ff_picks[region] = max(c, key=lambda t: ff_ev(t,ROUND_PTS[4]),
                               default=region_opts[region]['regional_champ'])

    seen={}
    for region,team in list(ff_picks.items()):
        if team in seen:
            alts=[t for t in champ_cands if t not in ff_picks.values()]
            ff_picks[region]=max(alts,key=lambda t:ff_ev(t,ROUND_PTS[4]),default=team)
        seen[ff_picks[region]]=region

    return {'champion':champion,'ff_picks':ff_picks,'champ_pair':champ_ff_pair,
            'pool_evs':{'champion':ff_ev(champion,ROUND_PTS[5]),
                        **{r:ff_ev(t,ROUND_PTS[4]) for r,t in ff_picks.items()}}}

# =============================================================================
# PRINTING
# =============================================================================
ROUND_PTS = [1,2,4,8,16,32]

def sep(t=''):
    print(f"\n{'='*68}")
    if t: print(f"  {t}")
    print(f"{'='*68}")

def print_summary(sim, chalk, ann, region_opts, ff_opt, bracket, protect):
    n = sim['n']
    sep("EXPECTED VALUE ANALYSIS")
    pr1p = chalk['jp_r1p']; pr12p = chalk['jp_r1r2p']
    ev1  = pr12p * ann['gross']
    ev2  = (pr1p - pr12p) * PRIZE_R1
    ev3  = PRIZE_POOL / POOL_SIZE
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

    sep("CHALK PICKS — R1 + R2")
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

    sep("FIVE-BRACKET POOL PICKS — S16 ONWARD")
    print(f"  Champion: {ff_opt['champion']}  "
          f"(sim={sim['ch_counts'].get(ff_opt['champion'],0)/n*100:.1f}%  "
          f"pool_ev={ff_opt['pool_evs']['champion']:.2f})\n")
    for region in REGIONS:
        opt = region_opts[region]; rc = sim['region_counts'][region]
        champ = opt['regional_champ']; pod = opt['pod_of_champ']
        top = opt['s16_top']; bot = opt['s16_bot']
        print(f"  [{region[:1]}] {region:<10} Regional champ: {champ:<20} "
              f"E8 sim={rc.get(champ,[0]*4)[3]/n*100:.1f}%  ev={opt['pool_evs']['e8']:.2f}")
        for label, pick, rnd_idx in [('S16 top',top,2),('S16 bot',bot,2)]:
            sp = rc.get(pick,[0]*4)[rnd_idx]/n
            flag = ' ← champ path' if pick==champ else ''
            print(f"       {label}: {pick:<20} sim={sp*100:.1f}%{flag}")

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

    sep("COMPLETE BRACKET — ALL ROUNDS")
    print("  R1+R2: chalk  |  S16+: pool-optimal\n")
    for region in REGIONS:
        opt=region_opts[region]; c=chalk[region]
        champ=opt['regional_champ']; rc=sim['region_counts'][region]
        print(f"  ═══ {region.upper()} ═══")
        for gi,(pick,wp,prot) in enumerate(zip(c['r1'],c['r1_wp'],c['r1_prot'])):
            tag=' [P]' if prot else ''
            print(f"  R1: {pick:<26}{wp*100:.0f}%{tag}")
        for pick,wp in zip(c['r2'],c['r2_wp']):
            print(f"  R2: {pick:<26}{wp*100:.0f}%")
        for pick,label in [(opt['s16_top'],'S16'),(opt['s16_bot'],'S16'),
                           (champ,'E8')]:
            sp=rc.get(pick,[0]*4)[2 if label=='S16' else 3]/n
            flag=' ← champ' if pick==champ else ''
            print(f"  {label}: {pick:<26}{sp*100:.0f}% sim{flag}")
        print()
    ff=ff_opt['ff_picks']; ch=ff_opt['champion']
    for region,team in ff.items():
        sp=sim['ff_counts'].get(team,0)/n
        flag=' ← champion' if team==ch else ''
        print(f"  FF {region:<10}: {team:<24}{sp*100:.0f}% sim{flag}")
    print(f"\n  CHAMPION: {ch}  ({sim['ch_counts'].get(ch,0)/n*100:.1f}% sim)")

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

    print("\nOptimizing five-bracket pool picks...")
    region_opts = {r: optimize_region(r,sim,bracket,chalk,teams,n) for r in REGIONS}
    for r,opt in region_opts.items():
        print(f"  {r}: {opt['regional_champ']}  (E8 sim {sim['region_counts'][r].get(opt['regional_champ'],[0]*4)[3]/n*100:.1f}%)")

    ff_opt = optimize_ff(sim, region_opts, teams, n)
    print(f"  Champion: {ff_opt['champion']}")

    print_summary(sim, chalk, ann, region_opts, ff_opt, bracket, PROTECT_1V16)

if __name__ == '__main__':
    main()
