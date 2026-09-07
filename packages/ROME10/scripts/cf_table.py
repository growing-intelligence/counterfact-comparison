"""CounterFact comparison table (Founder 2026-09-06): Method | Edits | Success | ΔMMLU-Pro | ΔGSM8K | ΔARC | ΔTQ | ΔGPQA | controls broken.
Success = edit prompts whose target_new is rank 1 over the full vocab after editing (same harness for all four). Δ = held-out
accuracy (pp) minus the pristine baseline measured once in the same regime. Reads /root/cf/results/<tag>_results.jsonl."""
import json, os
R = "/root/cf/results"; B = ["mmlu_pro", "gsm8k", "arc_challenge", "truthfulqa_mc2", "gpqa_diamond"]
bench = {int(json.loads(l)["idx"]): json.loads(l)["benchmark"] for l in open("/root/cf/eval/heldout_set.jsonl", encoding="utf-8")}
def ranks(tag):
    p = f"{R}/{tag}_results.jsonl"
    return {int(json.loads(l)["idx"]): json.loads(l)["rank"] for l in open(p)} if os.path.exists(p) else None
def acc(r, keys): return 100 * sum(1 for k in keys if r[k] == 1) / len(keys)
base = ranks("base_heldout"); base_c = ranks("base_controls"); rows = []
for m, tag in (("ROME", "ROME"), ("MEMIT", "MEMIT"), ("AlphaEdit", "AlphaEdit"), ("Probe56", "probe56")):
    e, h, c = ranks(f"{tag}_edits"), ranks(f"{tag}_heldout"), ranks(f"{tag}_controls")
    if e is None: rows.append({"method": m, "status": "not run"}); continue
    row = {"method": m, "edits": len(e), "success_pct": round(acc(e, list(e)), 1)}
    if h and base:
        for b in B: keys = [i for i in h if bench[i] == b]; row[f"d_{b}"] = round(acc(h, keys) - acc(base, keys), 2)
        row["d_total"] = round(acc(h, list(h)) - acc(base, list(base)), 2); row["heldout_regressed"] = sum(1 for i in h if base[i] == 1 and h[i] != 1); row["heldout_gained"] = sum(1 for i in h if base[i] != 1 and h[i] == 1)
    if c and base_c: row["controls_broken_above_floor"] = sum(1 for i in c if base_c[i] == 1 and c[i] != 1); row["controls_floor"] = sum(1 for i in base_c if base_c[i] != 1)
    rows.append(row)
out = {"baseline_heldout_pct": round(acc(base, list(base)), 2) if base else None, "rows": rows, "method": "rank-1 of the gold token over the full vocab; bf16 backbone, fp32 lm_head, LEFT padding, batch 24, deterministic; identical for all four methods"}
json.dump(out, open(f"{R}/cf_table.json", "w"), indent=1)
print(f"{'Method':10s} {'Edits':>5s} {'Success':>8s} " + " ".join(f"{'Δ'+b[:7]:>9s}" for b in B) + f" {'Δtotal':>7s} {'ctrl>floor':>10s}")
for r in rows:
    if "success_pct" not in r: print(f"{r['method']:10s} not run"); continue
    print(f"{r['method']:10s} {r['edits']:5d} {r['success_pct']:7.1f}% " + " ".join(f"{r.get('d_'+b, float('nan')):+9.2f}" for b in B) + f" {r.get('d_total', float('nan')):+7.2f} {r.get('controls_broken_above_floor', '-'):>10}")
print("CF_TABLE_DONE")
