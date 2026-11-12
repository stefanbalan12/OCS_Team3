import numpy as np, matplotlib.pyplot as plt
from experiment import run_one
from cec05 import make_cec05

D, fname = 10, 'F2'           # the function you choose
F, (lo, hi) = make_cec05(D)
MAXFE = 1000 * D              # matches your Oracle budget (handout says 10000 for D=10)

plt.figure(figsize=(8, 5))
for alg in ('NM', 'PRS', 'LRS', 'SW'):
    T = []
    for seed in range(1, 26):
        _, _, tr = run_one(alg, F[fname], D, lo, hi, seed)
        tr = np.r_[tr, np.full(MAXFE - len(tr), tr[-1])]   # pad early-stopped runs
        T.append(np.maximum(tr, 1e-8))                     # floor for log scale
    T = np.array(T)
    x = np.arange(MAXFE)
    plt.plot(x, np.median(T, 0), label=alg)
    plt.fill_between(x, *np.percentile(T, [25, 75], 0), alpha=.2)

plt.yscale('log')
plt.xlabel('function evaluations'); plt.ylabel('best-so-far error')
plt.title(f'{fname}, D={D}, 25 runs'); plt.legend()
plt.savefig('convergence.png', dpi=150, bbox_inches='tight')
plt.show()