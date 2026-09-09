"""EasyEdit covariance statistics (mom2 of the MLP output-projection input) for several layers in ONE pass, written in EasyEdit's own cache
format and file names. Everything else is EasyEdit's own code: dataset (wikimedia/wikipedia 20231101.en), TokenizedDataset truncation
(min(context, batch_tokens)), length_collation(batch_tokens), tally() sampler (random_sample=1, sample_size N, batch_size 100), flatten_masked_batch,
SecondMoment.add in float32.
  compute:  python cf_stats_onepass_gptj.py compute --layers 3,4,5,6,7,8 --sample_size 100000 --stats_dir <CF>/stats
  verify:   python cf_stats_onepass_gptj.py verify  --layer 5 --sample_size 4 --out <CF>/results/cov_onepass_verification.json
            (runs EasyEdit's native layer_stats for that layer on the same samples, then the one-pass writer, and compares bit for bit)
GPTJ: parameterized copy of pod_sync/cf_stats_onepass.py. Changes marked "# GPTJ:":
  --model default from CF_MODEL; --rewrite_tmpl (default transformer.h.{}.mlp.fc_out, was model.layers.{}.mlp.down_proj); npos read like EasyEdit
  (n_positions first, then max_position_embeddings); default layers 3..8 (MEMIT/AlphaEdit gpt-j-6B.yaml); sys.path from CF_ROOT.
  For GPT-J batch_tokens (8192) >= npos (2048) so the file name carries NO '_t{batch_tokens}' suffix: <stats_dir>/<model basename>/wikipedia_stats/transformer.h.<L>.mlp.fc_out_float32_mom2_100000.npz
  The statistic is over fc_out's 16384-dim input: each file is a 16384x16384 float32 matrix (~1.07 GB), vs 14336^2 (822 MB) for Llama."""
import argparse, json, os, sys, time, hashlib
CF = os.environ.get("CF_ROOT", "/workspace/cf"); sys.path.insert(0, f"{CF}/EasyEdit")                                                   # GPTJ: was /root/cf/EasyEdit
import numpy as np, torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyeditor.models.rome import layer_stats as LS
from easyeditor.models.rome.tok_dataset import TokenizedDataset, dict_to_, flatten_masked_batch, length_collation
from easyeditor.util.runningstats import CombinedStat, SecondMoment, tally, save_cached_state
from datasets import load_dataset
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["compute", "verify"]); ap.add_argument("--layers", default="3,4,5,6,7,8"); ap.add_argument("--layer", type=int, default=5)   # GPTJ: default layers 3-8
ap.add_argument("--sample_size", type=int, default=100000); ap.add_argument("--batch_tokens", type=int, default=int(os.environ.get("CF_STATS_BT", "8192"))); ap.add_argument("--stats_dir", default=f"{CF}/stats"); ap.add_argument("--out", default=f"{CF}/results/cov_onepass_verification.json")
ap.add_argument("--model", default=os.environ.get("CF_MODEL", "/workspace/gpt-j-6b")); ap.add_argument("--rewrite_tmpl", default=os.environ.get("CF_REWRITE_TMPL", "transformer.h.{}.mlp.fc_out")); A = ap.parse_args()   # GPTJ: module template parameter
BATCH_SIZE = 100; PRECISION = "float32"; DS = "wikipedia"
tok = AutoTokenizer.from_pretrained(A.model); model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="cuda"); model.eval()
for p in model.parameters(): p.requires_grad_(False)
model_name = model.config._name_or_path.rsplit("/")[-1]
def npos_of(cfg):                                                                                                                        # GPTJ: EasyEdit's own attribute order (layer_stats.py:136-143)
    for k in ("n_positions", "max_sequence_length", "max_position_embeddings", "seq_length"):
        if hasattr(cfg, k): return getattr(cfg, k)
    raise NotImplementedError
def fname(stats_dir, layer, sample_size, batch_tokens):
    npos = npos_of(model.config); size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos: size_suffix = "_t{batch_tokens}" + size_suffix          # EasyEdit's literal (unformatted) suffix, reproduced on purpose (absent for GPT-J: 8192 >= 2048)
    return f"{stats_dir}/{model_name}/{DS}_stats/{A.rewrite_tmpl.format(layer)}_{PRECISION}_mom2{size_suffix}.npz"                    # GPTJ: template instead of model.layers.{}.mlp.down_proj
def get_ds(batch_tokens):
    raw_ds = load_dataset("wikimedia/wikipedia", "20231101.en"); maxlen = npos_of(model.config)                                      # GPTJ: npos_of
    if batch_tokens is not None and batch_tokens < maxlen: maxlen = batch_tokens
    return TokenizedDataset(raw_ds["train"], tok, maxlen=maxlen)
class _Stop(Exception): pass
def onepass(layers, sample_size, batch_tokens, stats_dir):
    ds = get_ds(batch_tokens); names = {L: A.rewrite_tmpl.format(L) for L in layers}; stats = {L: CombinedStat(mom2=SecondMoment()) for L in layers}   # GPTJ: template
    cap = {}; handles = []
    for L in layers:
        mod = model.get_submodule(names[L]); handles.append(mod.register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, i[0].detach())))   # input of fc_out = the fitted-statistic tensor (retain_input)
    last = max(layers); handles.append(model.get_submodule(names[last]).register_forward_hook(lambda m, i, o: (_ for _ in ()).throw(_Stop())))   # stop after the last traced layer, as Trace(stop=True) does
    loader = tally(CombinedStat(mom2=SecondMoment()), ds, cache=None, sample_size=sample_size, batch_size=BATCH_SIZE, collate_fn=length_collation(batch_tokens), pin_memory=True, random_sample=1, num_workers=2)
    n_tok = 0; t0 = time.time()
    with torch.no_grad():
        for batch_group in tqdm(loader, total=-(-sample_size // BATCH_SIZE)):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                try: model(**batch)
                except _Stop: pass
                for L in layers:
                    feats = flatten_masked_batch(cap[L], batch["attention_mask"]).to(dtype=torch.float32); stats[L].add(feats)
                n_tok += int(batch["attention_mask"].sum())
    for h in handles: h.remove()
    out = {}
    for L in layers:
        stats[L].to_(device="cpu"); fn = fname(stats_dir, L, sample_size, batch_tokens); os.makedirs(os.path.dirname(fn), exist_ok=True)
        save_cached_state(fn, stats[L], {"sample_size": sample_size}); out[L] = fn
    print(f"[onepass] layers {layers}: {sample_size} samples, {n_tok:,} tokens, {time.time()-t0:.0f}s -> {list(out.values())}", flush=True); return out, n_tok
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
if A.mode == "compute":
    layers = [int(x) for x in A.layers.split(",")]; out, n_tok = onepass(layers, A.sample_size, A.batch_tokens, A.stats_dir)
    json.dump({"layers": layers, "sample_size": A.sample_size, "batch_tokens": A.batch_tokens, "tokens": n_tok, "rewrite_tmpl": A.rewrite_tmpl, "files": {str(L): {"path": p, "sha256": sha(p)} for L, p in out.items()}}, open(f"{A.stats_dir}/onepass_manifest.json", "w"), indent=1)
    print("ONEPASS_DONE", flush=True)
else:
    # 1) EasyEdit's own writer for one layer on the same small sample (its code path, untouched), into a separate stats dir
    nat_dir = f"{CF}/stats_verify_native"; one_dir = f"{CF}/stats_verify_onepass"; lname = A.rewrite_tmpl.format(A.layer)                 # GPTJ: CF_ROOT + template
    class H: device = 0
    t0 = time.time(); LS.layer_stats(model, tok, lname, nat_dir, DS, to_collect=["mom2"], sample_size=A.sample_size, precision=PRECISION, batch_tokens=A.batch_tokens, hparams=H()); t_nat = time.time() - t0
    nat = fname(nat_dir, A.layer, A.sample_size, A.batch_tokens); assert os.path.exists(nat), nat
    # 2) the one-pass writer, same samples, same cap, all covariance layers (the verified layer among them)
    out, n_tok = onepass([int(x) for x in A.layers.split(",")], A.sample_size, A.batch_tokens, one_dir); one = out[A.layer]               # GPTJ: layers from --layers (was the literal [4,5,6,7,8])
    a, b = np.load(nat), np.load(one)
    keys = sorted(set(a.files) | set(b.files)); rep = {"layer": A.layer, "sample_size": A.sample_size, "tokens": n_tok, "batch_tokens": A.batch_tokens, "native_file": nat, "onepass_file": one, "native_seconds": round(t_nat, 1), "keys": {}}
    identical = True
    for k in keys:
        if k not in a.files or k not in b.files: rep["keys"][k] = "MISSING"; identical = False; continue
        x, y = a[k], b[k]
        if x.dtype.kind in "iu" or x.shape == (): eq = bool(np.array_equal(x, y)); rep["keys"][k] = {"equal": eq, "native": str(x), "onepass": str(y)}
        else: eq = bool(np.array_equal(x, y)); rep["keys"][k] = {"equal": eq, "shape": list(x.shape), "dtype": str(x.dtype), "max_abs_diff": float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))), "native_sha256_bytes": hashlib.sha256(x.tobytes()).hexdigest(), "onepass_sha256_bytes": hashlib.sha256(y.tobytes()).hexdigest()}
        identical &= eq
    rep["bit_identical"] = identical; rep["native_file_sha256"] = sha(nat); rep["onepass_file_sha256"] = sha(one)
    json.dump(rep, open(A.out, "w"), indent=1); print(json.dumps({k: v for k, v in rep.items() if k != "keys"}, indent=0)); print("VERIFY_PASS" if identical else "VERIFY_FAIL", flush=True)
