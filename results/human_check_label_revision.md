# Human-check label revision (2026-10-06)

You re-uploaded `eval/human_check_questions.csv` and `eval/human_check_answers.csv` after noticing mistakes. Only the label column changed (same ids, same order, identical question/answer/context text). This file records what changed relative to the labels committed in `eee4a95`; the new labels supersede the old ones everywhere.

## Questions: 8 of 20 labels changed (old count valid 19 / ambiguous 1 / invalid 0  ->  new valid 11 / ambiguous 7 / invalid 2)

| id | type | old | new |
|---|---|---|---|
| ts006 | single | valid | ambiguous |
| tm007 | multi | valid | invalid |
| ts033 | single | valid | ambiguous |
| ts043 | single | valid | invalid |
| ts023 | single | valid | ambiguous |
| ts051 | single | valid | ambiguous |
| ts001 | single | valid | ambiguous |
| tm001 | multi | valid | ambiguous |

## Answers: 5 labels changed meaning, 1 changed wording only

| id | old | new |
|---|---|---|
| tm002 | not faithful | faithful |
| tm010 | faithful | not faithful |
| ts078 | faithful | not faithful |
| ts073 | faithful | not faithful |
| ts006 | not faithful | faithful |

Wording only (same meaning): tm015: 'Unfaithful' -> 'not faithful'

## Effect on the headline human-check numbers

| statistic | before | after |
|---|---|---|
| questions valid / ambiguous / invalid | 19 / 1 / 0 | 11 / 7 / 2 |
| judge vs human faithfulness: percent agreement | 80% (16/20) | 85% (17/20) |
| judge vs human faithfulness: Cohen's kappa | 0.216 | 0.483 |
| human faithful / not faithful | 17 / 3 | 16 / 4 |
| items where the judge was more lenient / stricter than the human | 2 / 2 | 2 / 1 |

The frozen test question set was NOT edited: questions you now label ambiguous or invalid stay in every headline retrieval and answer metric, as the protocol requires (no edits to questions or gold labels after seeing results). They are a validity caveat, and an exploratory split by label is in `results/s2_exploratory_validity_split.md`.
