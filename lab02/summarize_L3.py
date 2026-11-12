import sys
import pandas as pd

# usage: python summarize.py [l2_results.csv | l3_results.csv]   (default: l2_results.csv)
path = sys.argv[1] if len(sys.argv) > 1 else 'l2_results.csv'
df = pd.read_csv(path)

med = df.groupby(['D','func','alg']).error.median().unstack()
succ = (df.assign(ok=df.error < 1e-8)
          .groupby(['D','func','alg']).ok.sum().unstack())

print(med.to_string(float_format=lambda v: f'{v:.3g}'))
print(succ)
print(df.groupby(['alg','D']).fe.median())

# Lab 3, Task 2: median error of LRS at D=10 as a function of the step size
try:
    sg = pd.read_csv('l3_sigma.csv')
    print('\nTask 2: median error of LRS (D=10) vs sigma')
    print(sg.groupby(['func', 'sigma']).error.median().unstack()
            .to_string(float_format=lambda v: f'{v:.3g}'))
except FileNotFoundError:
    pass