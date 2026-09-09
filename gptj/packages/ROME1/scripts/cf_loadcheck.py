"""Cross-version load check: score the first 48 controls with the edited ROME model in THIS transformers version (venv 5.5.4) and
with the base model, same regime as the harness (bf16, left padding, padding-aware position_ids, gold rank over the full vocab)."""
import json, sys, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
rows = [json.loads(l) for l in open("/root/cf/data/cf_controls_500.jsonl")][:48]
tok = AutoTokenizer.from_pretrained("/root/llama-3.1-8b"); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
def score(md):
    m = AutoModelForCausalLM.from_pretrained(md, dtype=torch.bfloat16, device_map="cuda"); m.eval(); ranks = []
    with torch.no_grad():
        for s in range(0, len(rows), 24):
            b = rows[s:s+24]; enc = tok([r["prompt"] for r in b], return_tensors="pt", padding=True).to("cuda"); pos = (enc.attention_mask.cumsum(-1)-1).clamp(min=0)
            lg = m(**enc, position_ids=pos, use_cache=False).logits[:, -1, :].float()
            ranks += [int((lg[i] > lg[i, r["gold_token_id"]]).sum()) + 1 for i, r in enumerate(b)]
    del m; torch.cuda.empty_cache(); return ranks
print("transformers", transformers.__version__)
for md in sys.argv[1:]:
    r = score(md); print(md, "rank1", sum(1 for x in r if x == 1), "/", len(r), "ranks[:8]", r[:8], flush=True)
