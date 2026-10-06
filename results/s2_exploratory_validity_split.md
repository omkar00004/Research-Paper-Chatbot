# EXPLORATORY: retrieval metrics by human validity label

Added after the human-check labels were revised. The frozen test set is unchanged and every headline number still uses all 120 answerable test questions. Groups are tiny and mix single- and multi-passage questions (multi-passage Hit@k needs ALL gold spans, so groups with more multi-passage questions score lower for that reason alone); treat this as a diagnostic, not evidence. 'not audited' = the other test questions whose validity nobody checked.

| group | n (multi-passage) | cond | Hit@5 | nDCG@5 |
|---|---|---|---|---|
| valid | 11 (0) | C1 | 0.455 [0.182, 0.727] | 0.297 [0.093, 0.504] |
| valid | 11 (0) | C2 | 0.545 [0.273, 0.818] | 0.510 [0.238, 0.783] |
| valid | 11 (0) | C3 | 0.636 [0.364, 0.909] | 0.545 [0.273, 0.804] |
| valid | 11 (0) | S | 0.545 [0.273, 0.818] | 0.466 [0.182, 0.727] |
| ambiguous or invalid | 9 (3) | C1 | 0.333 [0.000, 0.667] | 0.253 [0.088, 0.437] |
| ambiguous or invalid | 9 (3) | C2 | 0.444 [0.111, 0.778] | 0.415 [0.177, 0.666] |
| ambiguous or invalid | 9 (3) | C3 | 0.667 [0.333, 0.889] | 0.551 [0.281, 0.812] |
| ambiguous or invalid | 9 (3) | S | 0.444 [0.111, 0.778] | 0.383 [0.139, 0.648] |
| not audited (reference) | 100 (17) | C1 | 0.540 [0.440, 0.640] | 0.445 [0.368, 0.525] |
| not audited (reference) | 100 (17) | C2 | 0.680 [0.590, 0.770] | 0.586 [0.508, 0.665] |
| not audited (reference) | 100 (17) | C3 | 0.670 [0.580, 0.760] | 0.608 [0.531, 0.687] |
| not audited (reference) | 100 (17) | S | 0.620 [0.520, 0.710] | 0.587 [0.500, 0.671] |

## Paired C2 - C1 within each group (absolute metric points, 95% bootstrap CI)

| group | n (multi-passage) | Hit@5 | nDCG@5 |
|---|---|---|---|
| valid | 11 (0) | +0.091 [-0.182, +0.364] | +0.214 [-0.016, +0.486] |
| ambiguous or invalid | 9 (3) | +0.111 [+0.000, +0.333] | +0.162 [-0.025, +0.404] |
| not audited (reference) | 100 (17) | +0.140 [+0.050, +0.230] | +0.142 [+0.070, +0.214] |
