import pandas as pd
df = pd.read_csv('l2_results.csv')

med = df.groupby(['D','func','alg']).error.median().unstack()
succ = (df.assign(ok=df.error < 1e-8)
          .groupby(['D','func','alg']).ok.sum().unstack())

print(med.to_string(float_format=lambda v: f'{v:.3g}'))
print(succ)
print(df.groupby(['alg','D']).fe.median())