import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

df = pd.read_csv("l2_results.csv")
med = df.groupby(["D", "func", "alg"]).error.median().unstack()

ranks = med.rank(axis=1)
print(ranks.mean())

print(friedmanchisquare(med["NM"], med["PRS"], med["LRS"]))

p = {a: wilcoxon(med["LRS"], med[a]).pvalue for a in ("NM", "PRS")}
k = len(p)
prev = 0

for i, (a, pv) in enumerate(sorted(p.items(), key=lambda t: t[1])):
    adj = max(prev, min(1.0, (k - i) * pv))
    prev = adj
    print(a, round(pv, 5), "Holm-adjusted", round(adj, 5))
