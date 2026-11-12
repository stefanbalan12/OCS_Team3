import numpy as np, pandas as pd
from oracle import Oracle
from scipy.optimize import minimize
from cec05 import make_cec05 #your code for CEC2005 or the
from rs import (
    pure_random_search,
    localized_random_search,
    solis_wets_search,
) # implementations of RS, PRS, and Solis-Wets

import time, sys



def nm(o, rng): #nedler-mead
    x0 = rng.uniform(o.lb, o.ub)
    try:
        minimize(o, x0, method='Nelder-Mead',
                options={'maxfev': o.max_fe, 'xatol': 1e-12, 'fatol': 1e-12})
    except StopIteration:
        pass
ALGS = {'NM': nm,
        'PRS': pure_random_search,
        'LRS': lambda o, r: localized_random_search(o, r, sigma=0.05),
        'SW': solis_wets_search}

# Lab 3, Task 1: PRS, LRS (default sigma = 0.1) and Solis-Wets
ALGS3 = {'PRS': pure_random_search,
         'LRS': localized_random_search,      # sigma = 0.1 (fraction of the box width)
         'SW': solis_wets_search}

def run_one(alg, f, D, lo, hi, seed, algs=ALGS, fe_factor=1000):
    o = Oracle(f, [lo] * D, [hi] * D, max_fe=fe_factor * D)
    algs[alg](o, np.random.default_rng(seed))
    return o.best_f, o.fe, o.trace


def task1_lab3(fe_factor=1000):
    """Lab 3, Task 1: PRS vs LRS vs SW, D in {5, 10}, 25 seeds -> l3_results.csv"""
    rows = []
    for D in (5, 10):
        F, (lo, hi) = make_cec05(D)
        for fname, f in F.items():
            for alg in ALGS3:
                for seed in range(1, 26):
                    err, fe, _ = run_one(alg, f, D, lo, hi, seed, ALGS3, fe_factor)
                    rows.append((alg, fname, D, seed, err, fe))
    df = pd.DataFrame(rows, columns=['alg', 'func', 'D', 'run', 'error', 'fe'])
    df.to_csv('l3_results.csv', index=False)
    return df


def task2_lab3(sigmas=(0.01, 0.05, 0.1, 0.3), funcs=('F1', 'F3'), D=10, fe_factor=1000):
    """Lab 3, Task 2: step size of LRS on F1 and F3, D=10 -> l3_sigma.csv"""
    F, (lo, hi) = make_cec05(D)
    algs = {}
    rows = []
    for s in sigmas:
        algs[s] = lambda o, r, s=s: localized_random_search(o, r, sigma=s)   # s=s binds the loop value
    for fname in funcs:
        for s in sigmas:
            for seed in range(1, 26):
                err, fe, _ = run_one(s, F[fname], D, lo, hi, seed, algs, fe_factor)
                rows.append((s, fname, D, seed, err, fe))
    df = pd.DataFrame(rows, columns=['sigma', 'func', 'D', 'run', 'error', 'fe'])
    df.to_csv('l3_sigma.csv', index=False)
    return df

def lab2():
    start = time.perf_counter()
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

    elapsed = time.perf_counter() - start
    print(f"Runtime: {elapsed:.2f} s")


if __name__ == '__main__':
    # usage: python experiment.py [lab2 | lab3 | all]   (default: task1 + task2)
    what = sys.argv[1] if len(sys.argv) > 1 else 'lab3'
    start = time.perf_counter()
    if what in ('lab2', 'all'):
        lab2()
    if what in ('lab3', 'all'):
        task1_lab3()
        task2_lab3()
        
    print(f"Runtime: {time.perf_counter() - start:.2f} s")