# Releases — choose your alpha

Two **different** abliteration recipes. Pick by use case.

| | **v1.0 alpha** | **v1.1 alpha** (this work) |
|---|---|---|
| **Goal** | Max refusal bypass / “full” ablit | **Mida / Brikie / Hermes-friendly** (less skill spill) |
| **Weights recipe** | L10–42 `wo_b` **+ MTP** · λ=3.5 · rank-1 SRA | L10–42 `wo_b` · **MTP stock** · λ=3.5 · rank-1 SRA |
| **Early layers** | L0–9 stock | L0–9 stock |
| **Refusal 32/32** | **100%** | **100%** |
| **C1 pure (code)** | **~57 tok/s** | **~50 tok/s** (slightly slower) |
| **Standing concurrency** | often C12 in 1M recipes | **C=4** recommended standing |
| **Agent / tools** | Needs ops (on-demand skills, LCM scrub) | **Milder protocol surface** (greetings cleaner; still use ops) |
| **HF (gated)** | [Abliterated-Uncensored](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored) | […-v1.1-alpha-Mida-Brikie](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-v1.1-alpha-Mida-Brikie) |
| **Serve default** | `scripts/dsv4-nvfp4-1m-serve.sh` (seqs=12) | `scripts/serve-v1.1-alpha.sh` (**seqs=4**, 1M) |

> **Not the same model.** v1.1 is not a config flag on v1.0 — it is a **different weight edit** (MTP left stock). Download the HF repo that matches the version you want.

Both releases are **gated** on Hugging Face: same **Username / Email / intended use** fields and **Responsible Use** checkboxes. See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

---

## v1.0 alpha — full abliteration

- **Publish name:** DeepSeek-V4-Flash-DSpark Abliterated (Uncensored) · 1M · ~57 tok/s  
- **Edit:** main L10–42 **and** `mtp.*.attn.wo_b`  
- **Best for:** maximum uncensored chat / refusal eval, max C1  
- **Tradeoff:** more Hermes skill-catalog spill under fat skill indexes and multi-tool (Mida/Brikie) loops unless ops are tight  

Docs: [RESULTS.md](RESULTS.md) · [docs/STATUS.md](docs/STATUS.md) · [docs/STATUS_FINETUNE.md](docs/STATUS_FINETUNE.md)

```bash
hf download drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored \
  --local-dir ~/models/dsv4-flash-dspark-abliterated
```

---

## v1.1 alpha — milder / Mida–Brikie friendly · C=4

- **Publish name:** DeepSeek-V4-Flash-DSpark Abliterated Uncensored **v1.1 alpha**  
- **Edit:** main L10–42 only · **MTP not abliterated**  
- **Best for:** Hermes / agent / multi-read (Mida/Brikie) with fewer initial skill leaks  
- **Tradeoff:** C1 pure **~50 tok/s** (vs ~57 for v1.0); standing serve **C=4** at 1M  
- **Still:** **32/32** refusal bypass · 0 garble on the publish battery  

Raw: [results/v1.1-alpha/](results/v1.1-alpha/) · [docs/STATUS_V1.1_ALPHA.md](docs/STATUS_V1.1_ALPHA.md)

### Measured (v1.1 · 2× GB10 TP=2 · stage-c)

| Metric | Value |
|---|---|
| Context | **1,048,576** |
| KV pool @ 1M / UTIL 0.85 | **~2.89M tokens** |
| Standing **max_num_seqs** | **4** |
| C1 server agg (code, ignore_eos) | **~50 tok/s** |
| C4 server agg | **~113 tok/s** (from C1–C12 sweep) |
| C12 server agg | **~216 tok/s** (optional; not standing) |
| DSpark accept (C1–C12) | **~56–63%** |
| Refusal suite | **32/32 = 100%** |

### Build v1.1 from stock

```bash
# Pristine stock required (do not project onto a dirty hardlinked tree)
python3 scripts/project_wob.py \
  --src ~/models/dsv4-flash-dspark \
  --dst ~/models/dsv4-flash-dspark-abliterated-mida \
  --direction results/refusal_direction_r1.pt \
  --lambda-attn 3.5 --min-layer 10 --max-layer 42 \
  --n-directions 1 --no-mtp
```

Or download **gated** weights (same Username / Email / intended-use gate as v1.0):

```bash
hf download drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-v1.1-alpha-Mida-Brikie \
  --local-dir ~/models/dsv4-flash-dspark-abliterated-mida
```

### Serve v1.1 (C=4 · 1M)

```bash
# Image
docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
docker tag ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10 \
  vllm-dspark-runtime:dspark-nvfp4-stage-c

# Rank1 then rank0 — edit MASTER/IF/HCA in script if needed
MODELDIR=~/models/dsv4-flash-dspark-abliterated-mida \
  CTX=1048576 SEQS=4 UTIL=0.85 \
  bash scripts/serve-v1.1-alpha.sh 1
MODELDIR=~/models/dsv4-flash-dspark-abliterated-mida \
  CTX=1048576 SEQS=4 UTIL=0.85 \
  bash scripts/serve-v1.1-alpha.sh 0
```

---

## Which should I use?

| If you need… | Choose |
|---|---|
| Fastest C1, max “hot” ablit | **v1.0 alpha** |
| Hermes / tools / Mida–Brikie with less spill | **v1.1 alpha** |
| Both | Run **two weight trees**; do not mix shards |

Ops still required for agents (names-only skills index, on-demand skills, LCM scrub). See [docs/HERMES_SPILL_FIX.md](docs/HERMES_SPILL_FIX.md).
