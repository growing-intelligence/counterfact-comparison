"""Agent C — loader verifier (Founder 2026-09-07). Verify only, never writes results. For every edited model and the base:
load it in EasyEdit's own environment (transformers 5.5.4, which reads the saved rope_parameters natively), score 200 random
controls (seed 42) in the harness regime (bf16, left padding, padding-aware position ids, batch 24, fp32 logits, rank of the
gold token over the full vocabulary), and compare rank-for-rank against the harness's controls file (transformers 4.55.2 with
the config translation applied). Output: results/agentC_loadverify_v2.json + one line per model."""
import json, random, sys, time, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
CF = "/workspace/cf"; M = sys.argv[1]; controls = [json.loads(l) for l in open(f"{CF}/data/cf_controls_500.jsonl", encoding="utf-8")]
sample = controls   # v2: ALL 500 controls in the harness order and batch composition (batches of 24), so numerics are apples to apples
tok = AutoTokenizer.from_pretrained("/workspace/cf/gpt-j-6b-bf16"); tok.padding_side = "left"; tok.truncation_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
def ranks(md):
    m = AutoModelForCausalLM.from_pretrained(md, dtype=torch.bfloat16, device_map="cuda"); m.eval(); out = {}
    with torch.no_grad():
        for s in range(0, len(sample), 24):
            b = sample[s:s + 24]; enc = tok([r["prompt"] for r in b], return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda"); pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
            lg = m(**enc, position_ids=pos, use_cache=False).logits[:, -1, :].float()
            for i, r in enumerate(b): out[r["idx"]] = int((lg[i] > lg[i, r["gold_token_id"]]).sum()) + 1
    rope = (float(m.config.rope_theta) if hasattr(m.config, "rope_theta") else None, getattr(m.config, "rope_scaling", None) or getattr(m.config, "rope_parameters", None))
    del m; torch.cuda.empty_cache(); return out, rope
rep = {"transformers": transformers.__version__, "controls_scored": 500, "batching": "harness order, batches of 24", "models": {}}
for name, md, tag in (("base", "/workspace/cf/gpt-j-6b-bf16", "base"), (M, f"{CF}/edited/{M}/model", M)):   # GPT-J: one edited model per call, always against the base
    t0 = time.time(); mine, rope = ranks(md)
    harness = {json.loads(l)["idx"]: json.loads(l)["rank"] for l in open(f"{CF}/results/{tag}_controls_results.jsonl")}
    match = sum(1 for i in mine if mine[i] == harness.get(i)); both1 = sum(1 for i in mine if (mine[i] == 1) == (harness.get(i) == 1))
    diffs = sorted([(i, mine[i], harness.get(i)) for i in mine if mine[i] != harness.get(i)], key=lambda x: -abs(x[1] - (x[2] or 0)))[:10]
    rep["models"][name] = {"rank_match": match, "rank1_agreement": both1, "of": 500, "rope_as_loaded": [rope[0], rope[1]], "largest_diffs_idx_mine_harness": diffs, "seconds": round(time.time() - t0, 1)}
    print(f"AGENT C {name}: ranks match {match}/500, rank-1 agreement {both1}/500, rope theta {rope[0]} type {(rope[1] or {}).get('rope_type')}; {time.time()-t0:.0f}s", flush=True)
json.dump(rep, open(f"{CF}/results/agentC_loadverify_{M}.json", "w"), indent=1); print("AGENTC_DONE", flush=True)
