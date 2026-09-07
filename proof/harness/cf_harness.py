"""cf_harness — open scoring harness for the CounterFact 1,000-edit comparison (Growing Intelligence, 2026-09).
Scoring only. No editing method and no Probe56 logic lives here.

Regime (identical for every method in the published comparison): bf16 weights on CUDA, LEFT padding with padding-aware
position ids, batch 24 in fixed file order, logits of the last prompt token cast to fp32, rank of the gold token over the
full vocabulary (rank 1 = correct). One results line per question: {"key","idx","source","benchmark","rank"}.

Library:
    model, tok = load_model(model_dir, base_dir)        # base_dir supplies the tokenizer and the reference rotary settings
    rows = load_split(data_dir, "edits"|"controls"|"heldout")
    summary = score_split(model, tok, rows, out_path, on_batch=None)
`on_batch(batch_rows, enc)` is an optional callback invoked after tokenization and before the forward pass; the harness
does nothing with it. It exists so a caller can attach its own per-batch bookkeeping without touching this file.

CLI (score an unmodified or edited model):
    python cf_harness.py --model <dir> --base <base_dir> --data <data_dir> --split heldout --out results/
"""
import argparse, hashlib, json, os, time
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BATCH = 24


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""): h.update(chunk)
    return h.hexdigest()


def _config_matching_base(model_dir, base_dir):
    """Models saved by a newer transformers may store rotary settings under `rope_parameters`; make sure the loaded config
    carries the same rope_theta / rope_scaling as the base model, whatever key layout the checkpoint uses."""
    raw = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8")); cfg = AutoConfig.from_pretrained(model_dir)
    rp = raw.get("rope_parameters")
    if rp and not raw.get("rope_scaling"):
        cfg.rope_scaling = {k: v for k, v in rp.items() if k != "rope_theta"}; cfg.rope_theta = float(rp.get("rope_theta", getattr(cfg, "rope_theta", 0.0)))
    if base_dir:
        base = json.load(open(os.path.join(base_dir, "config.json"), encoding="utf-8")); brs = base.get("rope_scaling") or {}; crs = dict(getattr(cfg, "rope_scaling", None) or {})
        assert float(getattr(cfg, "rope_theta", 0.0)) == float(base.get("rope_theta", 0.0)), f"rope_theta mismatch vs base: {getattr(cfg, 'rope_theta', None)} vs {base.get('rope_theta')}"
        for k in ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings", "rope_type"):
            assert crs.get(k) == brs.get(k), f"rope_scaling[{k}] mismatch vs base: {crs.get(k)} vs {brs.get(k)}"
    return cfg


def load_model(model_dir, base_dir=None, device="cuda"):
    tok = AutoTokenizer.from_pretrained(base_dir or model_dir); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=_config_matching_base(model_dir, base_dir), torch_dtype=torch.bfloat16, device_map=device); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, tok


def load_split(data_dir, split):
    name = {"edits": "cf_edits_1000.jsonl", "controls": "cf_controls_500.jsonl", "heldout": "heldout_5209.jsonl"}[split]; rows = []
    for line in open(os.path.join(data_dir, name), encoding="utf-8"):
        r = json.loads(line); r["idx"] = int(r["idx"]); r["gold_token_id"] = int(r["gold_token_id"]); r["_key"] = f"{split}_{r['idx']}"; r["_src"] = split
        r.setdefault("benchmark", "counterfact"); rows.append(r)
    expected = {"edits": 1000, "controls": 500, "heldout": 5209}[split]
    assert len(rows) == expected, f"{split}: {len(rows)} rows, expected {expected}"
    return rows


@torch.no_grad()
def _score_batch(model, tok, batch_rows, on_batch, out_fh):
    enc = tok([r["prompt"] for r in batch_rows], return_tensors="pt", padding=True).to(model.device)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)                           # padding-aware position ids for LEFT-padded batches
    if on_batch is not None: on_batch(batch_rows, enc)
    logits = model(**enc, position_ids=pos, use_cache=False).logits[:, -1, :].float(); ranks = {}
    for i, r in enumerate(batch_rows):
        rk = int((logits[i] > logits[i, r["gold_token_id"]]).sum()) + 1; ranks[r["_key"]] = rk
        out_fh.write(json.dumps({"key": r["_key"], "idx": r["idx"], "source": r["_src"], "benchmark": r["benchmark"], "rank": rk}) + "\n")
    return ranks


def score_split(model, tok, rows, out_path, on_batch=None, log=print, batch=BATCH):
    """Score `rows` in file order, batches of `batch`; writes one JSON line per question to out_path (overwritten)."""
    got = {}; t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fh:
        for s in range(0, len(rows), batch):
            cur = rows[s:s + batch]
            try: got.update(_score_batch(model, tok, cur, on_batch, fh))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for r in cur: got.update(_score_batch(model, tok, [r], on_batch, fh))
            fh.flush()
            if (s // batch) % 40 == 0 or s + batch >= len(rows): log(f"  {rows[0]['_src']}: {len(got)}/{len(rows)} ({len(got)/max(1e-9, time.time()-t0):.1f}/s)")
    assert len(got) == len(rows), f"scored {len(got)} of {len(rows)}"
    n1 = sum(1 for v in got.values() if v == 1)
    return {"split": rows[0]["_src"], "n": len(rows), "rank1": n1, "pct": round(100 * n1 / len(rows), 2), "results": out_path, "sha256": sha256_file(out_path)}


def main():
    ap = argparse.ArgumentParser(description="CounterFact scoring harness (scoring only)")
    ap.add_argument("--model", required=True); ap.add_argument("--base", default=None, help="base model dir (tokenizer + reference rotary settings); defaults to --model")
    ap.add_argument("--data", required=True); ap.add_argument("--split", required=True, choices=["edits", "controls", "heldout"]); ap.add_argument("--out", default="results"); ap.add_argument("--tag", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); tag = a.tag or os.path.basename(os.path.normpath(a.model))
    model, tok = load_model(a.model, a.base or a.model); rows = load_split(a.data, a.split)
    s = score_split(model, tok, rows, os.path.join(a.out, f"{tag}_{a.split}_results.jsonl"))
    json.dump(s, open(os.path.join(a.out, f"{tag}_{a.split}_summary.json"), "w"), indent=1); print(json.dumps(s))


if __name__ == "__main__": main()
