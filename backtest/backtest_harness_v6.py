"""
March Madness Backtester v6 — Full Enhancement
===============================================
New in v6 (Tier 1 + Tier 2):

TIER 1:
  1. Bayesian seed-matchup prior blend
     p_final = α * p_model + (1-α) * p_prior_35yr
     α varies by matchup: 0.30 (1v16) → 0.85 (8v9)
     Addresses residual overconfidence in 85-95% bucket.

  2. Conference champion flag: binary → power / mid-major split
     power_conf_champ: ACC/B1G/B12/SEC/BigEast = larger β (elite confirmation)
     mid_conf_champ:   all others = small β (hot team signal only)

TIER 2:
  3. Round-specific sigmoid temperature
     T_R1=1.0, T_R2=0.92, T_S16=0.84, T_E8=0.76, T_FF=0.70, T_Champ=0.70
     Later rounds compress extreme WP — surviving teams are more evenly matched.

  4. Student-t chaos epsilon (df=4) instead of Normal
     Same center and scale, but 2.6x more probability mass in extreme tails.
     Matches historical frequency of chaos years (2022 style) better.

  5. KenPom × Q1_Momentum interaction term
     β_interact * (kenpom_log_a * mom_log_a - kenpom_log_b * mom_log_b)
     Captures "dominant AND peaking" compound effect.
     Note: one new beta in the vector; near-zero if data is sparse.

All v5 features retained: log KenPom/CoachXP/SOS/Mom, locked injury β=-0.244,
portal-corrected seniority, phase-aware tournament momentum.
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from model_core import (
    CORPUS_INJURY_BETA, ROUND_TEMP, SEED_PRIORS, SEED_ALPHA,
    POWER_CHAMPS, MID_CHAMPS, POWER_CONFS,
    log_norm, log_rank,
    get_seed_prior, bayesian_blend,
    tempered_sig, chaos_eps,
    vegas_coverage, injury_residual,
    get_champ_flags,
)
# Backwards-compat aliases (harness internals used these names)
tempered_sigmoid = tempered_sig
chaos_epsilon    = chaos_eps

np.random.seed(42)

COINFLIP_BAND      = (0.44, 0.56)
GD_STEPS           = 100_000
ROUND_ORDER        = ['R1','R2','S16','E8','FF','Champ']


def kenpom_mom_interaction(team, teams, momentum_val):
    """Product of log-KenPom advantage × log-momentum for interaction term."""
    kp = teams.get(team,{}).get('kenpom_rank', 100)
    # Higher rank = weaker; invert so "strength" is increasing
    kp_strength = log_norm(max(0, 200 - kp), scale=200)
    mom_strength = log_norm(momentum_val, scale=10)
    return kp_strength * mom_strength

# =============================================================================
# REMAINING UNCHANGED FROM v5: injury, phase momentum, data loading
# =============================================================================

    return max(min(1.0, sev/2.5), min(1.0, abs(sharp)/2.0))
    return sev * (1.0 - vegas_coverage(sev, sharp))

def build_tournament_records(results):
    cumulative, snapshot = {}, {}
    for rnd in ROUND_ORDER:
        for _, row in results[results['round']==rnd].iterrows():
            for t in [str(row['team_a']), str(row['team_b'])]:
                snapshot.setdefault(t, {})
                if rnd not in snapshot[t]:
                    snapshot[t][rnd] = dict(cumulative.get(t, {'wins':0,'games':0}))
        for _, row in results[results['round']==rnd].iterrows():
            w = str(row['winner'])
            for t in [str(row['team_a']), str(row['team_b'])]:
                cumulative.setdefault(t, {'wins':0,'games':0})
                cumulative[t]['games'] += 1
                if t == w: cumulative[t]['wins'] += 1
    return snapshot

def phase_momentum(team, rnd, base, rec):
    wins, games = rec.get('wins',0), rec.get('games',0)
    if rnd == 'R1': return base
    if rnd == 'R2':
        r1 = 10.0 if (games>=1 and wins>=1) else 0.0
        return 0.6*(base*0.8) + 0.4*r1
    return (wins/games*10.0) if games>0 else base

def load_year(year):
    _here = Path(__file__).parent
    base = _here.parent / f"backtest/data/{year}"
    missing = [str(base/f) for f in
               [f"teams_{year}.csv",f"vegas_{year}.csv",f"results_{year}.csv"]
               if not (base/f).exists()]
    if missing: raise FileNotFoundError(f"\n  Missing: {missing}")
    teams   = pd.read_csv(base/f"teams_{year}.csv").set_index('team').to_dict('index')
    vdf     = pd.read_csv(base/f"vegas_{year}.csv")
    vegas   = [(r['team_a'],r['team_b'],r['p_a_wins'],r['sharp_move_pts'])
               for _,r in vdf.iterrows()]
    sharp   = {(r['team_a'],r['team_b']): r['sharp_move_pts'] for _,r in vdf.iterrows()}
    sharp.update({(r['team_b'],r['team_a']): -r['sharp_move_pts'] for _,r in vdf.iterrows()})
    results = pd.read_csv(base/f"results_{year}.csv")
    return teams, vegas, results, sharp

# =============================================================================
# v6 MODEL — 9 FEATURES (8 calibrated + locked injury), TEMPERED SIGMOID
# =============================================================================

def factors_v6(a, b, teams, year, mp_a=None, mp_b=None, mt_a=None, mt_b=None):
    ta, tb = teams.get(a,{}), teams.get(b,{})
    if not ta or not tb: return np.zeros(9)

    ra, rb = ta.get('kenpom_rank',100), tb.get('kenpom_rank',100)
    kenpom = log_rank(rb) - log_rank(ra)

    mp_a_ = mp_a if mp_a is not None else ta.get('momentum',5.0)
    mp_b_ = mp_b if mp_b is not None else tb.get('momentum',5.0)
    mt_a_ = mt_a if mt_a is not None else 0.0
    mt_b_ = mt_b if mt_b is not None else 0.0

    pca, mca = get_champ_flags(a, year)
    pcb, mcb = get_champ_flags(b, year)

    # Interaction: (dominant + hot) compound effect
    interact_a = kenpom_mom_interaction(a, teams, mp_a_)
    interact_b = kenpom_mom_interaction(b, teams, mp_b_)

    return np.array([
        kenpom,                                                    # log KenPom
        (ta['seniority']   - tb['seniority'])   / 10.0,           # linear
        (log_norm(ta['coach_xp']) - log_norm(tb['coach_xp'])),     # log
        (log_norm(ta['sos_score'])- log_norm(tb['sos_score'])),     # log
        (log_norm(mp_a_)   - log_norm(mp_b_)),                     # log pre-mom
        (mt_a_             - mt_b_)             / 10.0,            # linear tourney-mom
        (pca - pcb),                                               # power conf champ
        (mca - mcb),                                               # mid conf champ
        (interact_a        - interact_b),                          # KenPom×Mom
    ])

def base_logit(beta, a, b, teams, year, mp_a=None, mp_b=None,
               mt_a=None, mt_b=None):
    f = factors_v6(a, b, teams, year, mp_a, mp_b, mt_a, mt_b)
    return beta[0] + np.dot(beta[1:], f)

def predict_v6(beta, a, b, teams, year, rnd='R1',
               mp_a=None, mp_b=None, mt_a=None, mt_b=None,
               ir_a=0.0, ir_b=0.0,
               seed_a=None, seed_b=None):
    if not a or not b or a==b: return 0.99
    logit = base_logit(beta, a, b, teams, year, mp_a, mp_b, mt_a, mt_b)
    # Locked injury adjustment
    # ir_a - ir_b: opponent injury helps team A (positive delta → higher logit)
    inj_delta = (ir_a - ir_b) / 3.0
    logit    += CORPUS_INJURY_BETA * 3.0 * inj_delta
    # Tempered sigmoid (round-specific temperature)
    p_model = tempered_sigmoid(logit, rnd)
    # Bayesian seed-matchup blend (R1 only — priors are R1 base rates)
    if rnd == 'R1' and seed_a is not None:
        p_model = bayesian_blend(p_model, seed_a, seed_b)
    return p_model

def calibrate_v6(vegas, teams, year, n_steps=GD_STEPS, verbose=True):
    """
    10-weight vector (9 features + bias). Injury locked, not calibrated.
    R1 calibration data: apply Bayesian blend to calibration targets too,
    so model learns residuals after prior adjustment.
    """
    beta = np.array([0.0, 2.5, 0.40, 0.30, 0.50, 0.30, 0.0, 0.25, 0.10, 0.05])
    for step in range(n_steps):
        lr = 0.05 if step<40000 else 0.02 if step<70000 else 0.005
        grad = np.zeros(10)
        for a, b, vp, sm in vegas:
            ta = teams.get(a,{}); tb = teams.get(b,{})
            sa, sb = ta.get('seed'), tb.get('seed')
            f  = factors_v6(a, b, teams, year, mt_a=0.0, mt_b=0.0)
            logit = beta[0] + np.dot(beta[1:], f)
            p_raw = tempered_sigmoid(logit, 'R1')
            p_blended = bayesian_blend(p_raw, sa, sb)
            e = p_blended - vp
            # Gradient through blend: de/dlogit = alpha * sigmoid_deriv
            prior, alpha = get_seed_prior(sa, sb)
            alpha = alpha if alpha else 1.0
            dsig  = p_raw * (1.0 - p_raw) / ROUND_TEMP.get('R1', 1.0)
            grad[0]  += e * alpha * dsig
            grad[1:] += e * alpha * dsig * f
        beta -= lr * grad / max(len(vegas),1)
        if verbose and (step+1) % 25000 == 0:
            loss = sum(-(vp*np.log(np.clip(
                bayesian_blend(tempered_sigmoid(
                    beta[0]+np.dot(beta[1:],
                    factors_v6(a,b,teams,year,mt_a=0,mt_b=0)),'R1'),
                    teams.get(a,{}).get('seed'), teams.get(b,{}).get('seed')),
                1e-9,1-1e-9))+(1-vp)*np.log(np.clip(1-
                bayesian_blend(tempered_sigmoid(
                    beta[0]+np.dot(beta[1:],
                    factors_v6(a,b,teams,year,mt_a=0,mt_b=0)),'R1'),
                    teams.get(a,{}).get('seed'), teams.get(b,{}).get('seed')),
                1e-9,1-1e-9))) for a,b,vp,sm in vegas)/len(vegas)
            print(f"  Step {step+1:>6,} | Loss: {loss:.5f}")
    return beta

# =============================================================================
# VALIDATION RUNNER
# =============================================================================

def run_validation(year, teams, vegas, results, sharp, coinflip_only=False):
    print(f"\n  Calibrating v6 on {year} Vegas lines ({len(vegas)} games)...")
    beta = calibrate_v6(vegas, teams, year, GD_STEPS, verbose=True)

    t_records = build_tournament_records(results)
    predictions, mismatches = [], []

    for _, row in results.iterrows():
        a, b   = str(row['team_a']), str(row['team_b'])
        winner = str(row['winner'])
        rnd    = str(row['round'])
        if a not in teams or b not in teams:
            mismatches.append((rnd,a,b)); continue

        sa = teams[a].get('seed'); sb = teams[b].get('seed')
        sm_a = sharp.get((a,b), 0.0)
        ir_a = injury_residual(teams[a].get('injury_severity',0), sm_a) \
               if rnd=='R1' else teams[a].get('injury_severity',0)*0.3
        ir_b = injury_residual(teams[b].get('injury_severity',0),-sm_a) \
               if rnd=='R1' else teams[b].get('injury_severity',0)*0.3

        rec_a = t_records.get(a,{}).get(rnd,{'wins':0,'games':0})
        rec_b = t_records.get(b,{}).get(rnd,{'wins':0,'games':0})
        mp_a  = phase_momentum(a,rnd,teams[a].get('momentum',5.0),rec_a)
        mp_b  = phase_momentum(b,rnd,teams[b].get('momentum',5.0),rec_b)
        mt_a  = rec_a.get('wins',0)/rec_a.get('games',1)*10 \
                if rec_a.get('games',0)>0 else 0.0
        mt_b  = rec_b.get('wins',0)/rec_b.get('games',1)*10 \
                if rec_b.get('games',0)>0 else 0.0

        wp = predict_v6(beta,a,b,teams,year,rnd=rnd,
                        mp_a=mp_a,mp_b=mp_b,mt_a=mt_a,mt_b=mt_b,
                        ir_a=ir_a,ir_b=ir_b,seed_a=sa,seed_b=sb)
        predictions.append((wp, 1 if winner==a else 0, rnd, a, b, winner))

    return beta, predictions, mismatches

# =============================================================================
# REPORTING
# =============================================================================

def brier(p): return np.mean([(x-o)**2 for x,o,*_ in p]) if p else None

def sep(t=''):
    print(f"\n{'='*68}")
    if t: print(f"  {t}")
    print(f"{'='*68}")

def print_report(year, beta, preds, mismatches, coinflip_only=False):
    label = f"{year} {'COIN-FLIP' if coinflip_only else 'FULL'} — v6 (Tier1+2)"
    sep(label)

    if mismatches:
        print(f"  Missing: {[f'{r}:{a}v{b}' for r,a,b in mismatches[:4]]}")

    if coinflip_only:
        lo,hi = COINFLIP_BAND
        preds = [(p,o,r,a,b,w) for p,o,r,a,b,w in preds if lo<=p<=hi]
        print(f"\n  Coin-flip [{lo:.0%}–{hi:.0%}]: {len(preds)} games")

    if not preds: print("  No predictions."); return

    bs  = brier(preds)
    acc = np.mean([1 if (p>=0.5)==(o==1) else 0 for p,o,*_ in preds])
    print(f"\n  Games: {len(preds)}  Accuracy: {acc*100:.1f}%  Brier: {bs:.4f}")

    BUCKETS = [(0.44,0.56),(0.56,0.65),(0.65,0.75),(0.75,0.85),(0.85,0.95),(0.95,1.0)]
    print(f"\n  {'Bucket':<12} {'N':>4}  {'Pred%':>7}  {'Act%':>7}  {'Gap':>7}  Status")
    print(f"  {'-'*52}")
    for lo,hi in BUCKETS:
        b = [(p,o) for p,o,*_ in preds if lo<=p<hi]
        if not b: continue
        ap,ao = np.mean([p for p,o in b]), np.mean([o for p,o in b])
        gap   = ap - ao
        st    = "OK" if abs(gap)<0.04 else ("over" if gap>0 else "under")
        print(f"  {lo:.0%}-{hi:.0%}     {len(b):>4}  {ap*100:>6.1f}%  "
              f"{ao*100:>6.1f}%  {gap*100:>+6.1f}%  {st}")

    lo,hi = COINFLIP_BAND
    flips = [(p,o,r,a,b,w) for p,o,r,a,b,w in preds if lo<=p<=hi]
    if flips:
        cf = sum(1 for p,o,*_ in flips if (p>=0.5)==(o==1))
        print(f"\n  Coin-flip: {cf}/{len(flips)} correct ({cf/len(flips)*100:.0f}%)")
        print(f"\n  {'Rnd':<5} {'Team A':<20} {'Team B':<20} {'WP':>6}  "
              f"{'Winner':<20} OK")
        print(f"  {'-'*76}")
        for p,o,r,a,b,w in sorted(flips, key=lambda x:x[0]):
            ok = "YES" if (p>=0.5)==(o==1) else "NO"
            print(f"  {r:<5} {a:<20} {b:<20} {p*100:>5.1f}%  {w:<20} {ok}")

    if not coinflip_only:
        print(f"\n  By round:")
        rg = {}
        for p,o,r,*_ in preds: rg.setdefault(r,[]).append((p,o))
        for rnd in ROUND_ORDER:
            if rnd not in rg: continue
            g = rg[rnd]
            a = np.mean([1 if (p>=0.5)==(o==1) else 0 for p,o in g])
            b = np.mean([(p-o)**2 for p,o in g])
            print(f"    {rnd:<7}: {len(g):>2} games  acc={a*100:.0f}%  brier={b:.4f}")

    labels = ['bias','KenPom_log','Seniority','CoachXP_log','SOS_log',
              'Mom_pre_log','Mom_tourney','PowerCC','MidCC','KpxMom']
    print(f"\n  Beta weights (injury LOCKED={CORPUS_INJURY_BETA}):")
    for lbl,b in zip(labels,beta):
        print(f"    {lbl:<14}: {b:>+8.4f}")
    print(f"    {'InjResid':<14}: {CORPUS_INJURY_BETA:>+8.4f}  [LOCKED]")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',     type=int, required=True)
    parser.add_argument('--coinflip', action='store_true')
    args = parser.parse_args()
    print(f"\n  Backtester v6 (Tier1+2) | {args.year} | "
          f"{'coin-flip' if args.coinflip else 'full'}")
    print("  Bayes prior · power/mid CC · sigmoid temp · t-chaos · interaction\n")
    try:
        teams,vegas,results,sharp = load_year(args.year)
    except FileNotFoundError as e:
        print(e); return
    beta,preds,mismatches = run_validation(
        args.year,teams,vegas,results,sharp,args.coinflip)
    print_report(args.year,beta,preds,mismatches,args.coinflip)

if __name__ == '__main__':
    main()
