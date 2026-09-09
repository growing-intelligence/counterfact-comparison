# CounterFact × 4 Methods — Two Models, One Harness

Same four methods, the same 1,000 CounterFact edits, the same scoring harness, tested on two unrelated architectures — **Llama-3.1-8B** (Meta) and **GPT-J-6B** (EleutherAI). Different builders, different tokenizers, different internals, years apart.

**Same pattern.**

| Model | Params | Layers | Hidden | Vocab |
|---|---|---|---|---|
| Llama-3.1-8B | 8,030,261,248 | 32 | 4,096 | 128,256 |
| GPT-J-6B | 6,050,882,784 | 28 | 4,096 | 50,400 |

Parameter counts are summed tensor-by-tensor from the pinned weight files themselves, not quoted from a model card: Llama over the four `model-0000N-of-00004.safetensors` shards whose sha256 each match the certificate's `weights.Base.files` (291 tensors, all parameters); GPT-J over `pytorch_model.bin` sha256 `0e183edc2025ecfd…` at revision `47e169305d…`, counting the 285 parameter tensors and excluding 56 non-parameter buffers (the 2,048×2,048 causal masks and `masked_bias` scalars, 117,440,540 elements) that are registered on the module but are not weights.

Layers, hidden size and vocabulary are read from each model's own `config.json` at the revision pinned in that model's signed certificate — Llama `LlamaForCausalLM`, `config.json` sha256 `54acfad3cffe0576…` (32 attention heads, 8 key-value heads, grouped-query); GPT-J `GPTJForCausalLM`, `config.json` sha256 `9328fc6e157f344f…` at revision `47e169305d2e8376be1d31e765533382721b2cc1` (16 attention heads, rotary dim 64). This is the raw HuggingFace download's config; the certificate's `Base` is a re-serialized bf16 rebuild with a different `config.json` hash (`ff6d3b54…`) for the same pinned revision. The two share a hidden size and nothing else: different depth, different attention shape, and a vocabulary two and a half times larger on one than the other.

---

## The numbers, side by side

A method is scored on three things: how many of the 1,000 edits it actually landed, how many **nearby facts** — facts in the same domain that the model already answered correctly — it broke on the way, and what it did to 5,209 held-out general-benchmark questions.

| Method | Llama-3.1-8B edits fixed | Llama nearby broken | GPT-J-6B edits fixed | GPT-J nearby broken |
|---|---|---|---|---|
| ROME | 7 / 1,000 | 492 / 492 | 0 / 1,000 | 499 / 499 |
| MEMIT | 916 / 1,000 | 48 / 492 | 990 / 1,000 | 110 / 499 |
| AlphaEdit | 990 / 1,000 | 202 / 492 | 987 / 1,000 | 143 / 499 |
| **Probe56** | 841 / 1,000 | **0 / 492** | 985 / 1,000 | **0 / 499** |

Held-out general benchmarks, same runs:

| Method | Llama-3.1-8B | GPT-J-6B |
|---|---|---|
| ROME | −33.12 pp | −17.85 pp |
| MEMIT | −0.10 pp | +0.13 pp |
| AlphaEdit | −1.96 pp | −0.15 pp |
| **Probe56** | **0.00 pp** | **0.00 pp** |

The nearby-fact denominators differ (492 and 499) because each is the count of controls the *unmodified* model already answered at rank 1. Everything is counted against that floor, per model.

---

## Two conclusions

**1. Probe56 caused zero collateral damage on both models — the only method that did.**

Not "low." Zero, on a census of every control, on both architectures. 0 of 492 on Llama-3.1-8B and 0 of 499 on GPT-J-6B, with held-out accuracy bit-identical to the unmodified model in both cases (1,753/5,209 and 930/5,209 — the same rows, not merely the same total). Every other method broke nearby facts on both models. The weights are byte-identical to the frozen base in both runs; the SHA256 in each certificate is the base model's own.

**2. Probe56 is the only method that shows what happened inside the model at all.**

Every other method here is a black box that sometimes breaks nearby facts, with no way to see which ones or why until you go and check by hand. You edit, and then you discover the damage — or you don't discover it, because general benchmarks don't look where the damage is. Probe56 reports what it touched, per fact, before you ask. That visibility is the difference between a method you can audit and a method you can only test.

---

## What varied between the models

The two runs do not agree on how much damage each method does.

MEMIT is the clean one on Llama — 48 of 492 nearby facts broken, 9.8% — and more than twice as damaging on GPT-J: 110 of 499, 22.0%. AlphaEdit runs the other way: 202 of 492 on Llama (41.1%), 143 of 499 on GPT-J (28.7%). On Llama, MEMIT is the safe choice and AlphaEdit the risky one; on GPT-J that ordering narrows to almost nothing. ROME collapses totally on both, but the collapse on GPT-J is *degenerate* — most of its broken controls resolve to a single stuck token rather than to varied wrong answers, so there is barely a wrong answer left to inspect; on Llama the wreckage at least still produces distinguishable garbage.

**The damage rate is unpredictable across architectures, and that is itself the finding: you cannot assume a method is safe on your model because it looked safe on someone else's.** A published collateral number is a measurement of one method on one model, not a property of the method. If you are editing a model, the only number that tells you anything about your model is the one you measure on it.

---

## Full detail per model

- [`llama/`](llama/) — Llama-3.1-8B: report, five reproducibility packages, signed certificate `GIC-COUNTERFACT-2026-001`, collateral tables
- [`gptj/`](gptj/) — GPT-J-6B: report, five reproducibility packages, signed certificate `GIC-COUNTERFACT-GPTJ-2026-001`, collateral tables

Each subfolder is self-contained: `report/` (HTML + PDF + publication figure), `packages/` (per-method edited-weight manifests, raw per-row results, EasyEdit hyperparameters and commit), `proof/` (signed certificate, Merkle root, SHA256SUMS, `verify_certificate.py`), `collateral_examples/` (every broken control, with the decoded wrong answer the edited model now gives).

---

## Standing on their shoulders

This comparison exists because of four groups who built the field and opened their work:

**Meng, Bau, Andonian, Belinkov** — ROME (NeurIPS 2022) and MEMIT (ICLR 2023). They showed factual knowledge can be located and edited. They created CounterFact. Every method here, including ours, is measured on their benchmark.
https://arxiv.org/abs/2202.05262 · https://arxiv.org/abs/2210.07229

**Fang, Jiang, Wang, Ma, Jie, Wang, He, Chua** — AlphaEdit (ICLR 2025 Outstanding Paper). Null-space projection is elegant, and it works: 990/1,000 edits on Llama, 987/1,000 on GPT-J. It is the strongest weight-editing method we measured.
https://arxiv.org/abs/2410.02355

**Gu et al. (EMNLP 2024), Yang et al. (ACL 2024)** — They warned that editing harms general abilities. They were right, on both models.
https://arxiv.org/abs/2401.04700 · https://arxiv.org/abs/2402.09656

**EasyEdit team (Zhang et al.)** — One framework, all methods, reproducible. We ran their editing code unchanged, with the run options set explicitly per each paper's protocol (ROME sequential single edits; MEMIT one mass edit of 1,000; AlphaEdit batches of 100) and the covariance statistics computed once in their own cache format, verified against their writer.
https://github.com/zjunlp/EasyEdit

---

## A different approach

Probe56 does not edit weights. It adds runtime hooks that fire only on the question they were fitted for. On Llama-3.1-8B: 841 of 1,000 fixed, 0 broken. On GPT-J-6B: 985 of 1,000 fixed, 0 broken.

Fewer edits landed than AlphaEdit on Llama (84.1% vs 99.0%); slightly fewer on GPT-J (98.5% vs 98.7%). Zero collateral on both, verified by census rather than by sample.

**Every model today is patched by changing it. We show that a model can be corrected without being changed.**

---

## Verify it yourself

Every number in this repository is recounted from raw per-row result files. Each model ships five reproducibility packages, a signed certificate and a verifier:

```
python proof/verify_certificate.py CounterFact-certificate-SIGNED.json
```

which checks the Ed25519 signature, the certificate's own canonical hash, every file hash, and the Merkle root over them.

Growing Intelligence public key: `FtrWshUc/9rg5Cz+ARi5DP/yyqFhWMJLmx0VHKc3wpk=`

The Probe56 packages — encrypted plugin, frozen runner, open harness — are hosted on HuggingFace: https://huggingface.co/growing-intelligence/probe56-counterfact

---

Growing Intelligence · Pardes Hanna, Israel · growing-intelligence.com
Method patent-pending (USPTO 64/029,741, 64/048,143). Results public. Method private.
