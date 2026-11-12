import numpy as np
from scipy.optimize import minimize
from oracle import Oracle
from cec05 import make_cec05


def nm_restart(o, rng):
    try:
        while not o.exhausted:
            minimize(
                o,
                rng.uniform(o.lb, o.ub),
                method="Nelder-Mead",
                options={
                    "maxfev": o.max_fe - o.fe,
                    "xatol": 1e-12,
                    "fatol": 1e-12,
                },
            )
    except StopIteration:
        pass


if __name__ == "__main__":
    D = 10
    seed = 3
    F, (lo, hi) = make_cec05(D)

    o = Oracle(F["F1"], [lo] * D, [hi] * D, max_fe=10000 * D)
    rng = np.random.default_rng(seed)

    try:
        minimize(
            o,
            rng.uniform(o.lb, o.ub),
            method="Nelder-Mead",
            options={"maxfev": o.max_fe, "xatol": 1e-12, "fatol": 1e-12},
        )
    except StopIteration:
        pass

    print(np.round(o.best_x, 2))
