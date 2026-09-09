"""CounterFact controls (2026-09-06): controls must be rank 1 on the unmodified model (controls = correct questions). Pool = 4,000 CounterFact cases disjoint from the 1,000 edits (same seed-42 order), gold = target_true;
scored with cf_harness (same regime); the first 500 that are rank 1 become cf_controls_500.jsonl. Manifest updated."""
import json, subprocess, sys, hashlib
from tokenizers import Tokenizer
D = "/root/cf/data"; tok = Tokenizer.from_file("/root/cf/tokenizer/tokenizer.json")
sys.path.insert(0, "/root/cf"); from build_cf_splits import row, cases, order   # same seed-42 order, same row builder
edit_ids = {cases[i]["case_id"] for i in order[:1000]}
pool = [row(cases[i], "target_true", "control", 100000 + k) for k, i in enumerate(order[1000:5000])]; assert not ({r["case_id"] for r in pool} & edit_ids)
with open(f"{D}/cf_controls_pool.jsonl", "w", encoding="utf-8") as fh:
    for r in pool: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
subprocess.run([sys.executable, "-u", "/root/cf/cf_harness.py", "base_pool", "/root/llama-3.1-8b", "--split", "pool"], check=True, stdout=open("/root/cf/harness_base_pool.log", "w"), stderr=subprocess.STDOUT)
ranks = {int(json.loads(l)["idx"]): json.loads(l)["rank"] for l in open("/root/cf/results/base_pool_results.jsonl")}
ok = [r for r in pool if ranks.get(r["idx"]) == 1]; print(f"pool {len(pool)}: rank-1 on base {len(ok)}")
controls = ok[:500]; assert len(controls) == 500
for k, r in enumerate(controls): r["idx"] = 1000 + k; r["split"] = "control"
with open(f"{D}/cf_controls_500.jsonl", "w", encoding="utf-8") as fh:
    for r in controls: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
man = json.load(open(f"{D}/CF_SPLITS.manifest.json")); man["cf_controls_500.jsonl"] = hashlib.sha256(open(f"{D}/cf_controls_500.jsonl", "rb").read()).hexdigest()
man["controls_rule"] = "500 CounterFact cases disjoint from the edits, drawn from a 4,000-case pool, kept only if target_true is rank 1 on the unmodified model in the harness regime (baseline-correct controls, KIT)"; man["controls_pool"] = len(pool); man["controls_pool_rank1"] = len(ok)
json.dump(man, open(f"{D}/CF_SPLITS.manifest.json", "w"), indent=1); print("CF_CONTROLS_DONE")
