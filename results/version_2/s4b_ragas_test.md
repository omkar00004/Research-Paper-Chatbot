# S4b: RAGAS on the test split (optional)

**n = 30 single-passage questions (fixed subset of the S3 test subset); the judge is a single LLM (nemotron-3-super-120b).** Means with 95% bootstrap CIs; NaN samples are excluded (never scored as 0).

| condition | metric | mean [95% CI] | n used | NaN excluded | status |
|---|---|---|---|---|---|
| C1 | context_precision | 0.6210 [0.5018, 0.7358] | 30 | 0 | ok |
| C1 | faithfulness | 0.7833 [0.6444, 0.9111] | 30 | 0 | ok |
| C2 | context_precision | 0.5849 [0.4707, 0.6942] | 30 | 0 | ok |
| C2 | faithfulness | 0.7733 [0.6355, 0.8944] | 30 | 0 | ok |

## Paired difference C2 - C1 (same questions with both scores present)

- context_precision: C2 0.5849 vs C1 0.6210; absolute -0.0361 [-0.1679, +0.1054]; relative -5.8% [-24.9%, +19.8%]; n pairs = 30
- faithfulness: C2 0.7733 vs C1 0.7833; absolute -0.0100 [-0.1722, +0.1556]; relative -1.3% [-20.5%, +22.6%]; n pairs = 30
