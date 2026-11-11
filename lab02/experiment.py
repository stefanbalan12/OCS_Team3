import numpy as np, pandas as pd
from oracle import Oracle
from scipy.optimize import minimize
from cec05 import make_cec05 #your code for CEC2005 or the
from rs import pure_random_search, localized_random_search #implementations of RS an PRS

def nm(o, rng): #nedler-mead
    x0 = rng.uniform(o.lb, o.ub)
    try:
        minimize(o, x0, method='Nelder-Mead',
                options={'maxfev': o.max_fe, 'xatol': 1e-12, 'fatol': 1e-12})
    except StopIteration:
        pass
ALGS = {'NM': nm,
        'PRS': pure_random_search,
        'LRS': lambda o, r: localized_random_search(o, r, sigma=0.05)}

def run_one(alg, f, D, lo, hi, seed):
    o = Oracle(f, [lo] * D, [hi] * D, max_fe=1000 * D)
    ALGS[alg](o, np.random.default_rng(seed))
    return o.best_f, o.fe, o.trace


rows = []
for D in (5, 10):
    F, (lo, hi) = make_cec05(D)
    for fname, f in F.items():
        for alg in ALGS:
            for seed in range(1, 26):
                err, fe, _ = run_one(alg, f, D, lo, hi, seed)
                rows.append((alg, fname, D, seed, err, fe))
df = pd.DataFrame(rows, columns=['alg', 'func', 'D', 'run', 'error', 'fe'])
df.to_csv('l2_results.csv', index=False)