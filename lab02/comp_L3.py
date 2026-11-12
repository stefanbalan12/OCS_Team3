import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# Lab 3, Task 1: statistics for PRS vs LRS vs SW
df = pd.read_csv("l3_results.csv")
med = df.groupby(["D", "func", "alg"]).error.median().unstack()

ranks = med.rank(axis=1)
print(ranks.mean())

print(friedmanchisquare(med["SW"], med["LRS"], med["PRS"]))

# SW against each of the others, Holm correction
p = {a: wilcoxon(med["SW"], med[a]).pvalue for a in ("PRS", "LRS")}
k = len(p)
prev = 0

for i, (a, pv) in enumerate(sorted(p.items(), key=lambda t: t[1])):
    adj = max(prev, min(1.0, (k - i) * pv))
    prev = adj
    print("SW vs", a, round(pv, 5), "Holm-adjusted", round(adj, 5))