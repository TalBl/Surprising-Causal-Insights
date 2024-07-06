import pandas as pd
from algorithms.divexplorer.divexplorer import DivergenceExplorer

df = pd.read_csv('acstest.csv')
fp_diver = DivergenceExplorer(df)
subgroups = fp_diver.get_pattern_divergence(min_support=0.1, quantitative_outcomes=["Total person's income"])
subgroups = subgroups.sort_values(by="Total person's income_div", ascending=False, ignore_index=True)
d = subgroups.head(10)
d.to_csv("divexplorer_res.csv", index=False)
