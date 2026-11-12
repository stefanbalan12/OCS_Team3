import numpy as np

def pure_random_search(o, rng):
    try:
        while True:
            x = rng.uniform(o.lb, o.ub)
            o(x)  
    except StopIteration:
        pass

def localized_random_search(o, rng, sigma=0.1):
    """x_new = x + sigma*(u-l)*N(0,I); sigma is a fraction of the box width."""
    try:
        x = rng.uniform(o.lb, o.ub)
        current_val = o(x)

        step = (o.ub - o.lb) * sigma

        while True:
            x_new = np.clip(x + step * rng.standard_normal(x.shape), o.lb, o.ub)
            val = o(x_new)
            if val < current_val:
                x = x_new
                current_val = val
    except StopIteration:
        pass


def solis_wets_search(
    o,
    rng,
    step_size=0.1,
    expansion=2.0,
    contraction=0.5,
    successes_before_expansion=5,
    failures_before_contraction=3,
    min_step=1e-8,
):
    """Solis-Wets: x + b + sigma*N(0,I), then the opposite point x - b - sigma*N(0,I).
    Step doubled after 5 successes, halved after 3 failures; restarts from a
    random point when the step becomes smaller than min_step."""
    span = o.ub - o.lb
    try:
        while True:                              # restart loop
            x = rng.uniform(o.lb, o.ub)
            current_val = o(x)
            delta = step_size * span
            bias = np.zeros_like(x)
            successes = failures = 0

            while delta.max() >= min_step:
                d = delta * rng.standard_normal(x.shape)
                x_new = np.clip(x + bias + d, o.lb, o.ub)
                val = o(x_new)

                if val >= current_val:           # try the opposite point
                    x_new = np.clip(x - bias - d, o.lb, o.ub)
                    val = o(x_new)

                if val < current_val:
                    bias = 0.2 * (x_new - x) + 0.4 * bias
                    x, current_val = x_new, val
                    successes += 1
                    failures = 0
                else:
                    bias = bias * 0.4
                    successes = 0
                    failures += 1

                if successes >= successes_before_expansion:
                    delta = delta * expansion
                    successes = 0
                elif failures >= failures_before_contraction:
                    delta = delta * contraction
                    failures = 0
    except StopIteration:
        pass