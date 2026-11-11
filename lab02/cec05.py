import numpy as np

def Shifted_Sphere(z):
    return np.sum(z**2)

def Shifted_Schwefel(z):
    return np.sum(np.cumsum(z)**2)

def Shifted_Rotated_High_Conditioned_Elliptic(z):
    D = len(z)
    if D == 1:
        coefficients = np.array([1.0])
    else:
        coefficients = (10**6) ** (np.arange(D) / (D - 1))
    return np.sum(coefficients * (z**2))

def Shifted_Schwefel_Noise(z, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    base_val = np.sum(np.cumsum(z)**2)
    noise = 1 + 0.4 * np.abs(rng.standard_normal())
    return base_val * noise

def Schwefel_Global_Optimum_On_Bounds(x, o):
    D = len(x)
    rng = np.random.default_rng(42) # seed fixat pentru consistență
    A = rng.integers(-50, 51, size=(D, D))
    B = A @ o
    return np.max(np.abs(A @ x - B))


def make_cec05(D, seed=42):
    rng = np.random.default_rng(seed)
    
    # shift vector
    o = rng.uniform(-80, 80, size=D)
    
    # orthogonal matrix for F3
    H = rng.standard_normal((D, D))
    Q, _ = np.linalg.qr(H)
    M = Q

    functions = {
        'F1': lambda x: Shifted_Sphere(x - o),
        'F2': lambda x: Shifted_Schwefel(x - o),
        'F3': lambda x: Shifted_Rotated_High_Conditioned_Elliptic((x - o) @ M.T),
        'F4': lambda x: Shifted_Schwefel_Noise(x - o, rng),
        'F5': lambda x: Schwefel_Global_Optimum_On_Bounds(x, o)
    }
    
    lo, hi = -100.0, 100.0
    return functions, (lo, hi)

if __name__ == "__main__":
    for D in (5, 10):
        print(f"\n Test CEC 2005 (D = {D})")
        F, (lo, hi) = make_cec05(D)
        
        for fname, f in F.items():
            # random point for searching in [-100, 100]^D
            x_rand = np.random.uniform(lo, hi, size=D)
            val_rand = f(x_rand)
            
            print(f"[{fname}](D={D}, x_random) = {val_rand:.4e}")
