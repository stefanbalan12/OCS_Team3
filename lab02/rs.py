import numpy as np

def pure_random_search(o, rng):
    try:
        while True:
            x = rng.uniform(o.lb, o.ub)
            o(x)  
    except StopIteration:
        pass

def localized_random_search(o, rng, sigma=0.05):
    try:
        x = rng.uniform(o.lb, o.ub)
        current_val = o(x)
        
        span = (o.ub - o.lb) * sigma
        
        while True:
            x_new = x + rng.uniform(-span, span, size=o.lb.shape)
            x_new = np.clip(x_new, o.lb, o.ub)  
            val = o(x_new)
            if val < current_val:
                x = x_new
                current_val = val
    except StopIteration:
        pass