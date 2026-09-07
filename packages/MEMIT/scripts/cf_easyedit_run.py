"""CounterFact baselines with EasyEdit (Founder 2026-09-06): ROME, MEMIT, AlphaEdit — their code, their shipped Llama-3-8B
defaults, no tuning. Protocol per paper: ROME = 1,000 single edits applied sequentially (cumulative); MEMIT = one mass edit of
all 1,000 (the MEMIT setting); AlphaEdit = sequential edits in batches of 100 (the AlphaEdit setting). Only the model path in the
yaml is changed. Output: /root/cf/edited/<METHOD>/ (save_pretrained, bf16) + metrics.json (EasyEdit's own rewrite/rephrase/
locality metrics + wall time). Collateral is then measured by cf_harness.py, identical for all four methods.
usage: python cf_easyedit_run.py ROME|MEMIT|AlphaEdit
"""
import json, os, sys, time, shutil, yaml
sys.path.insert(0, "/root/cf/EasyEdit")
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams, AlphaEditHyperParams
import functools, importlib
STATS_BT = 8192   # Founder 2026-09-06: EasyEdit layer_stats batch_tokens cap (default for Llama-3.1 = 131,072 = context -> 45 GiB OOM)
DISCLOSURE = "batch_tokens capped at 8,192 (EasyEdit default = full 131,072 context). Covariance is a token sum; result identical. Applied to ROME, MEMIT, AlphaEdit equally."
for _mn in ("easyeditor.models.rome.compute_u", "easyeditor.models.memit.memit_main", "easyeditor.models.alphaedit.AlphaEdit_main"):
    _m = importlib.import_module(_mn)
    if callable(getattr(_m, "layer_stats", None)): _m.layer_stats = functools.partial(_m.layer_stats, batch_tokens=STATS_BT)

METHOD = sys.argv[1]; CF = "/root/cf"; MODEL = os.environ.get("CF_MODEL", "/root/llama-3.1-8b"); CF_N = int(os.environ.get("CF_N", "1000")); OUT = f"{CF}/edited/{METHOD}" + ("" if CF_N == 1000 else str(CF_N)); os.makedirs(OUT, exist_ok=True)
SRC = {"ROME": "hparams/ROME/llama3-8b.yaml", "MEMIT": "hparams/MEMIT/llama3-8b.yaml", "AlphaEdit": "hparams/AlphaEdit/llama3-8b.yaml"}[METHOD]
HP = {"ROME": ROMEHyperParams, "MEMIT": MEMITHyperParams, "AlphaEdit": AlphaEditHyperParams}[METHOD]
y = yaml.safe_load(open(f"{CF}/EasyEdit/{SRC}")); y["model_name"] = MODEL; y["stats_dir"] = f"{CF}/stats"; y["device"] = 0
if METHOD == "AlphaEdit": y["P_loc"] = f"{CF}/null_space_project_llama31.pt"
BATCH = {"ROME": None, "MEMIT": 1000, "AlphaEdit": 100}[METHOD]   # Founder 2026-09-06: MEMIT = one mass edit of all 1,000; AlphaEdit = batches of 100; EasyEdit default batch_size=1 would silently make both sequential single edits
if BATCH: y["batch_size"] = BATCH
yp = f"{OUT}/hparams_used.yaml"; yaml.safe_dump(y, open(yp, "w")); hp = HP.from_hparams(yp)
req = json.load(open(f"{CF}/data/cf_edits_1000_easyedit.json", encoding="utf-8")); assert len(req["prompts"]) == 1000
cut = lambda v: v[:CF_N] if isinstance(v, list) else {kk: cut(vv) for kk, vv in v.items()}; req = {k: cut(v) for k, v in req.items()}; N = len(req["prompts"]); assert N == CF_N
editor = BaseEditor.from_hparams(hp)
t0 = time.time(); log = open(f"{OUT}/edit.log", "a")
def L(m): print(f"[{time.strftime('%H:%M:%S')}][{METHOD}] {m}", flush=True); log.write(m + "\n"); log.flush()
L(f"start: model {MODEL}; hparams {SRC}; N={N}")
kw = dict(prompts=req["prompts"], target_new=req["target_new"], ground_truth=req["ground_truth"], subject=req["subject"], rephrase_prompts=req["rephrase_prompts"], locality_inputs=req["locality_inputs"])
if METHOD == "ROME":
    metrics, model, _ = editor.edit(**kw, sequential_edit=True, verbose=False)             # 1,000 single edits, cumulative
elif METHOD == "MEMIT":
    metrics, model, _ = editor.batch_edit(**kw, sequential_edit=True, verbose=False)                              # one mass edit of 1,000
else:
    metrics = []; model = None
    for s in range(0, N, min(100, N)):                                                              # AlphaEdit: sequential batches of 100
        sub = {k: (v[s:s + 100] if isinstance(v, list) else {kk: {kkk: vvv[s:s + 100] for kkk, vvv in vv.items()} for kk, vv in v.items()}) for k, v in kw.items()}
        m, model, _ = editor.batch_edit(**sub, sequential_edit=True, verbose=False); metrics.extend(m); L(f"AlphaEdit batch {s // 100 + 1}/10 done, {time.time() - t0:.0f}s")
dt = time.time() - t0
def acc(key):
    vals = [x["post"][key] for x in metrics if key in x.get("post", {})]
    flat = [float(v) if not isinstance(v, list) else (sum(map(float, v)) / max(1, len(v))) for v in vals]
    return (sum(flat) / len(flat)) if flat else None
summary = {"method": METHOD, "n_edits": N, "protocol": {"ROME": "sequential single edits", "MEMIT": "one batch of 1000", "AlphaEdit": "sequential batches of 100"}[METHOD], "hparams": SRC, "model": MODEL,
           "wall_s": round(dt, 1), "easyedit_rewrite_acc": acc("rewrite_acc"), "easyedit_rephrase_acc": acc("rephrase_acc"), "n_metric_rows": len(metrics), "layer_stats_batch_tokens": STATS_BT, "batch_size": BATCH, "sequential_edit": True, "easyedit_deviation": DISCLOSURE}
json.dump({"summary": summary, "metrics": metrics}, open(f"{OUT}/metrics.json", "w"), indent=1, default=str)
L(f"edits done in {dt:.0f}s; EasyEdit rewrite_acc={summary['easyedit_rewrite_acc']} rephrase_acc={summary['easyedit_rephrase_acc']}; saving weights")
model = model or editor.model; model = model.to("cpu"); import torch; model = model.to(torch.bfloat16); model.save_pretrained(f"{OUT}/model", safe_serialization=True); editor.tok.save_pretrained(f"{OUT}/model")
L("saved -> " + f"{OUT}/model"); print("CF_EASYEDIT_DONE", flush=True)
