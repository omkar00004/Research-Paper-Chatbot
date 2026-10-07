# EXPLORATORY: C2 without vs with the source-diversity step

Added after seeing the S2 results; not a pre-planned comparison. Difference = (C2 without diversity) - (production C2), same questions, paired 95% bootstrap CI (10,000 resamples). Absolute = metric points; relative = difference / production C2.

| type | n | metric | no-diversity | production C2 | absolute diff [95% CI] | relative diff [95% CI] |
|---|---|---|---|---|---|---|
| single | 100 | hit@5 | 0.790 | 0.740 | +0.050 [+0.000, +0.100] | +6.8% [+0.0%, +15.2%] |
| single | 100 | ndcg@5 | 0.660 | 0.606 | +0.054 [+0.024, +0.087] | +9.0% [+3.9%, +15.4%] |
| multi | 20 | hit@5 | 0.200 | 0.200 | +0.000 [-0.150, +0.150] | +0.0% [-66.7%, +133.3%] |
| multi | 20 | ndcg@5 | 0.365 | 0.369 | -0.004 [-0.062, +0.059] | -1.1% [-17.1%, +18.4%] |
