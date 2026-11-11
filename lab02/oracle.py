import numpy as np

class Oracle:
 """Black-box wrapper: counts FE, handles bounds, records the best-so-far trace."""
 def __init__(self, f, lower, upper, max_fe, f_opt=0.0):
    self.f, self.lb, self.ub = f, np.asarray(lower, float), np.asarray(upper, float)
    self.max_fe, self.f_opt = max_fe, f_opt
    self.fe, self.best_f, self.best_x, self.trace = 0, np.inf, None, []
 
 def __call__(self, x):
    if self.fe >= self.max_fe:
        raise StopIteration('budget exhausted')
    x = np.clip(np.asarray(x, float), self.lb, self.ub) # repair: nearest bound
    y = float(self.f(x)); self.fe += 1
    if y < self.best_f:
         self.best_f, self.best_x = y, x.copy()
    self.trace.append(self.best_f - self.f_opt) # best-so-far error
    return y