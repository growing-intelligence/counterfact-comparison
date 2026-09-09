"""CounterFact baselines with EasyEdit (Founder 2026-09-06): ROME, MEMIT, AlphaEdit — their code, their shipped defaults, no tuning.
Protocol per paper: ROME = 1,000 single edits applied sequentially (cumulative); MEMIT = one mass edit of all 1,000 (the MEMIT setting);
AlphaEdit = sequential edits in batches of 100 (the AlphaEdit setting). Only the model path in the yaml is changed. Output: <CF>/edited/<METHOD>/
(save_pretrained, bf16) + metrics.json (EasyEdit's own rewrite/rephrase/locality metrics + wall time). Collateral is then measured by cf_harness.py,
identical for all four methods.
usage: python cf_easyedit_run_gptj.py ROME|MEMIT|AlphaEdit
GPTJ: parameterized copy of pod_sync/cf_easyedit_run.py. Every change is marked "# GPTJ:". Environment:
  CF_ROOT      working root (was /root/cf)                          default /workspace/cf
  CF_MODEL     model dir (was /root/llama-3.1-8b); MUST contain 'gpt-j-6b' in its name (AlphaEdit P allocation, AlphaEdit_main.py:63-71)
  CF_HPARAMS_<METHOD>  hparams yaml relative to <CF_ROOT>/EasyEdit (was hparams/<METHOD>/llama3-8b.yaml)   default hparams/<METHOD>/gpt-j-6B.yaml
  CF_PLOC      AlphaEdit projection file (was <CF>/null_space_project_llama31.pt)                             default <CF_ROOT>/null_space_project_gptj.pt
  CF_STATS_BT  layer_stats batch_tokens cap (was 8192, inert for GPT-J: n_positions 2048 < cap)              default 8192
  CF_N         number of edits (1000 = the run; 1 / 10 = ROME1 / ROME10 sanity)                              default 1000
"""
import json, os, sys, time, shutil, yaml
CF = os.environ.get("CF_ROOT", "/workspace/cf")                                                        # GPTJ: was the literal "/root/cf"
sys.path.insert(0, f"{CF}/EasyEdit")                                                                   # GPTJ: path from CF_ROOT
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams, AlphaEditHyperParams
import functools, importlib
STATS_BT = int(os.environ.get("CF_STATS_BT", "8192"))   # GPTJ: env-overridable; Founder 2026-09-06 cap (Llama default = 131,072 context -> 45 GiB OOM). GPT-J n_positions=2048 < 8192 -> no truncation, no file-name suffix (layer_stats.py:155-162)
DISCLOSURE = f"batch_tokens capped at {STATS_BT:,} (EasyEdit default = 3 x n_positions). Covariance is a token sum; result identical. Applied to ROME, MEMIT, AlphaEdit equally. Inert for GPT-J (n_positions 2048 < cap)."   # GPTJ: text reflects the GPT-J context
for _mn in ("easyeditor.models.rome.compute_u", "easyeditor.models.memit.memit_main", "easyeditor.models.alphaedit.AlphaEdit_main"):
    _m = importlib.import_module(_mn)
    if callable(getattr(_m, "layer_stats", None)): _m.layer_stats = functools.partial(_m.layer_stats, batch_tokens=STATS_BT)

METHOD = sys.argv[1]; MODEL = os.environ.get("CF_MODEL", "/workspace/gpt-j-6b"); CF_N = int(os.environ.get("CF_N", "1000")); OUT = f"{CF}/edited/{METHOD}" + ("" if CF_N == 1000 else str(CF_N)); os.makedirs(OUT, exist_ok=True)   # GPTJ: default model dir name carries 'gpt-j-6b'
assert "gpt-j-6b" in MODEL.lower(), f"AlphaEdit allocates P only for model names containing gpt-j-6b/llama/qwen (AlphaEdit_main.py:63-71); got {MODEL}"   # GPTJ: guard
SRC = {m: os.environ.get(f"CF_HPARAMS_{m}", f"hparams/{m}/gpt-j-6B.yaml") for m in ("ROME", "MEMIT", "AlphaEdit")}[METHOD]   # GPTJ: was hparams/<METHOD>/llama3-8b.yaml
HP = {"ROME": ROMEHyperParams, "MEMIT": MEMITHyperParams, "AlphaEdit": AlphaEditHyperParams}[METHOD]
y = yaml.safe_load(open(f"{CF}/EasyEdit/{SRC}")); y["model_name"] = MODEL; y["stats_dir"] = f"{CF}/stats"; y["device"] = 0
if METHOD == "AlphaEdit": y["P_loc"] = os.environ.get("CF_PLOC", f"{CF}/null_space_project_gptj.pt")   # GPTJ: was <CF>/null_space_project_llama31.pt
BATCH = {"ROME": None, "MEMIT": 1000, "AlphaEdit": 100}[METHOD]   # Founder 2026-09-06: MEMIT = one mass edit of all 1,000; AlphaEdit = batches of 100; EasyEdit default batch_size=1 would silently make both sequential single edits
if BATCH: y["batch_size"] = BATCH
yp = f"{OUT}/hparams_used.yaml"; yaml.safe_dump(y, open(yp, "w")); hp = HP.from_hparams(yp)
req = json.load(open(f"{CF}/data/cf_edits_1000_easyedit.json", encoding="utf-8")); assert len(req["prompts"]) == 1000
cut = lambda v: v[:CF_N] if isinstance(v, list) else {kk: cut(vv) for kk, vv in v.items()}; req = {k: cut(v) for k, v in req.items()}; N = len(req["prompts"]); assert N == CF_N
editor = BaseEditor.from_hparams(hp)
t0 = time.time(); log = open(f"{OUT}/edit.log", "a")
def L(m): print(f"[{time.strftime('%H:%M:%S')}][{METHOD}] {m}", flush=True); log.write(m + "\n"); log.flush()
L(f"start: model {MODEL}; hparams {SRC}; N={N}; model dtype as loaded by EasyEdit: {next(editor.model.parameters()).dtype}")   # GPTJ: record the dtype (fp16:false / absent -> float32, editor.py:115)
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
        if s == 0 and not os.path.exists(y["P_loc"]) and os.path.exists("null_space_project.pt"): shutil.copyfile("null_space_project.pt", y["P_loc"]); L(f"P copied to {y['P_loc']} after batch 1 (Llama run lesson 18:50: batches 2+ must load P, not recompute it)")   # GPTJ: automates the manual copy the Llama run needed
dt = time.time() - t0
def acc(key):
    vals = [x["post"][key] for x in metrics if key in x.get("post", {})]
    flat = [float(v) if not isinstance(v, list) else (sum(map(float, v)) / max(1, len(v))) for v in vals]
    return (sum(flat) / len(flat)) if flat else None
summary = {"method": METHOD, "n_edits": N, "protocol": {"ROME": "sequential single edits", "MEMIT": "one batch of 1000", "AlphaEdit": "sequential batches of 100"}[METHOD], "hparams": SRC, "model": MODEL,
           "wall_s": round(dt, 1), "easyedit_rewrite_acc": acc("rewrite_acc"), "easyedit_rephrase_acc": acc("rephrase_acc"), "n_metric_rows": len(metrics), "layer_stats_batch_tokens": STATS_BT, "batch_size": BATCH, "sequential_edit": True, "easyedit_deviation": DISCLOSURE,
           "easyedit_load_dtype": str(next(editor.model.parameters()).dtype), "saved_dtype": "bfloat16"}   # GPTJ: dtype path recorded in the manifest
json.dump({"summary": summary, "metrics": metrics}, open(f"{OUT}/metrics.json", "w"), indent=1, default=str)
L(f"edits done in {dt:.0f}s; EasyEdit rewrite_acc={summary['easyedit_rewrite_acc']} rephrase_acc={summary['easyedit_rephrase_acc']}; saving weights")
model = model or editor.model; model = model.to("cpu"); import torch; model = model.to(torch.bfloat16); model.save_pretrained(f"{OUT}/model", safe_serialization=True); editor.tok.save_pretrained(f"{OUT}/model")
L("saved -> " + f"{OUT}/model"); print("CF_EASYEDIT_DONE", flush=True)
