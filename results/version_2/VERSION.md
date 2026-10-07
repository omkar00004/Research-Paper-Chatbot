# Evaluation report (version 2)

- **Source**: commit `124c9b5` (2026-10-06). Every file here is extracted verbatim from that commit.
- **Human check**: questions 11 valid / 7 ambiguous / 2 invalid (single-passage 11 of 17 valid, multi-passage 0 of 3 valid); judge vs human faithfulness agreement 85% (17 of 20), Cohen's kappa 0.483. The labels are in `human_check/`.
- **Frozen test set**: the question set and gold labels were not edited; questions labelled ambiguous or invalid stay in every headline metric.
- **Cost ledger**: 1617 HTTP calls, including the 562 of the optional S4b stage.

## Folder contents
- `results.md`, `paste_back.txt`, `summary.json`, `per_question.csv`, `dev_grid.md`, `error_analysis.md`, `call_ledger.md`: the report and its tables.
- `s0_recon.md`, `s1_...` to `s5_...` `.json`, `s4b_ragas_test.*`: per-stage result files; `s2_exploratory_*`: post-hoc analyses (exploratory).
- `human_check_agreement.txt` and `human_check/`: the human-check labels and the statistics computed from them.
- `eval_README.md`: `eval/README.md` at this version (method, assumptions, results and limitations).
- `questions.jsonl`: the frozen question set.
