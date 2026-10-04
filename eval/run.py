"""Entry point: python -m eval.run <preflight|s0|s1|s2|s3|s4|s5|s6|report|all> [--dry-run] [--max-calls N] [--max-cost USD] [--extras]

Exit codes: 3 quota/outage (checkpointed; re-run the printed command), 4 budget reached, 5 frozen-model drift.
"""
import argparse, os, sys, time
if "--dry-run" in sys.argv: os.environ["EVAL_RUN"] = "dry"
from eval.common import CACHE, EVAL, RESULTS, read_json, write_json, EXIT_BUDGET, EXIT_DRIFT, EXIT_QUOTA, ANSWER_MODEL, JUDGE_MODEL, WRITER_MODEL, State
from eval import llm as L

DRY_N = 5
import logging; logging.disable(logging.INFO)   # production modules log every rerank at INFO

def preflight(llm):
    L.set_stage("preflight")
    st = llm.model_status(); ok = True
    for role, m in (("question-writer", WRITER_MODEL), ("answer", ANSWER_MODEL), ("judge", JUDGE_MODEL)):
        info = st.get(m)
        if not info or not info.get("available") or info.get("execution_status") == "exhausted":
            print(f"  [FAIL] {role} model '{m}': {'unknown' if not info else info.get('execution_status')}. Not substituting; pick another and re-freeze."); ok = False; continue
        r = llm.chat(m, [{"role": "user", "content": "Reply with exactly the word: pong"}], max_tokens=150)
        print(f"  [ok] {role:16s} requested={m:24s} returned={r.model:34s} finish={r.finish_reason} cached={r.cached}")
    if not ok: raise SystemExit(EXIT_QUOTA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--max-calls", type=int); ap.add_argument("--max-cost", type=float)
    ap.add_argument("--extras", action="store_true")
    a = ap.parse_args()
    state = State(CACHE / "state_dry.json" if a.dry_run else EVAL / "state.json")
    llm = L.LLM(a.max_calls, a.max_cost); limit = DRY_N if a.dry_run else None
    stages = ["s1", "s2", "s3", "s4", "s5", "s6", "report"] if a.stage == "all" else [a.stage]
    cmd = "python -m eval.run " + " ".join(sys.argv[1:])
    try:
        t0 = time.time()
        for s in stages:
            t = time.time(); print(f"\n=== {s} {'(dry run)' if a.dry_run else ''} ===")
            if s == "preflight": preflight(llm)
            elif s == "s0": from eval import s0_recon; s0_recon.run()
            elif s == "s1": from eval import s1_questions as m; rows, rep = m.run(llm, state, limit); print(rep["counts"])
            elif s == "s2": from eval import s2_retrieval as m; m.run(limit)
            elif s == "s3": from eval import s3_answers as m; m.run(llm, state, limit, a.extras)
            elif s == "s4": from eval import s4_ragas as m; m.run(llm, state, limit)
            elif s == "s5": from eval import s5_errors as m; m.run(llm, state, limit)
            elif s == "s6": from eval import s6_human as m; m.run(limit)
            elif s == "report": from eval import report as m; m.run(limit)
            else: raise SystemExit(f"unknown stage {s}")
            state.flush(); dt = time.time() - t; print(f"[{s}] {dt:.1f}s")
            if s != "report":
                tp = RESULTS / "raw" / "stage_times.json"; tt = read_json(tp, []); tt.append({"stage": s, "seconds": round(dt, 1), "run": L.RUN}); write_json(tp, tt)
            if s not in ("report",): L.render_ledger_md(); 
    except L.QuotaExit as e:
        state.flush(); L.render_ledger_md()
        print(f"\nQUOTA/OUTAGE on provider '{e.provider}': {e.reason}\nCheckpoint saved ({state.path.name}). Resume later with:\n    {cmd}"); sys.exit(EXIT_QUOTA)
    except L.BudgetExceeded as e:
        state.flush(); L.render_ledger_md(); print(f"\nBUDGET STOP: {e}\nPartial results saved. Resume with a larger budget:\n    {cmd}"); sys.exit(EXIT_BUDGET)
    except L.LLMError as e:
        state.flush(); L.render_ledger_md(); print(f"\nLLM ERROR: {e}\nCheckpoint saved. Re-run to resume:\n    {cmd}"); sys.exit(1)
    except L.ModelDrift as e:
        state.flush(); print(f"\nMODEL DRIFT: {e}"); sys.exit(EXIT_DRIFT)
    if a.stage == "all" or a.stage in ("s1", "s2", "s3", "s4", "s5", "s6"):
        print("\n" + L.render_ledger_md())

if __name__ == "__main__":
    main()
