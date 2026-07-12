# DeepSeek-V4-Flash-DSpark — Abliterated (Uncensored)

Two **different** alpha releases. **Choose deliberately** — they are different weight edits.

| | **[v1.0 alpha](#v10-alpha--full-abliteration)** | **[v1.1 alpha](#v11-alpha--mida--brikie--hermes-friendly)** |
|---|---|---|
| **Who for** | Max ablit / max C1 | **Mida / Brikie / Hermes** (less skill spill) |
| **MTP draft heads** | Abliterated | **Stock** |
| **C1 pure** | **~57 tok/s** | **~50 tok/s** (slightly slower) |
| **Standing concurrency** | often C12 @ 1M | **C=4 @ 1M** |
| **Refusal 32/32** | 100% | 100% |
| **HF (gated)** | [Abliterated-Uncensored](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored) | […-v1.1-alpha-Mida-Brikie](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-v1.1-alpha-Mida-Brikie) |

**Full chooser + rebuild notes:** → **[RELEASES.md](RELEASES.md)**

![C1 performance](charts/c1-performance-ladder.png)

---

## Responsible Use (required — both HF repos)

> **WARNING:** These models have had **safety refusals removed**. Useful for red-teaming / research / unfiltered local assistants — and **you** must supply your own guardrails.

**Both Hugging Face weight repos are gated (`gated=auto`)** with the **same** access form:

| Field | Notes |
|---|---|
| **Username** | May default to your HF username |
| **Email** | May default to your HF account email |
| **Reason for intended use** | e.g. red-teaming, evaluation, research, local assistant |

Plus agreement checkboxes for prohibited uses.

**Full agreement:** → **[RESPONSIBLE_USE.md](RESPONSIBLE_USE.md)**

### Prohibited uses (you must agree)

- Anything involving the **sexual exploitation or endangerment of minors**
- You must be **18 years of age or older** to use and download
- Harmful recipes / substance knowledge from your prompts = **your responsibility and accountability**
- Content promoting **self-harm or suicide**
- Material **illegal in your jurisdiction**, or **harassment / doxxing / fraud** against real people
- Any use prohibited by the **upstream DeepSeek license**

---

## v1.0 alpha — full abliteration

**HF:** https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored  

| | |
|---|---|
| Context | **1,048,576** (`nvfp4_ds_mla`) |
| C1 pure | **~57 tok/s** |
| Refusal | **~100%** (32-prompt + hard probe) |
| Weights | L0–9 stock · L10–42 **+ MTP** ablit · λ=3.5 |

```bash
hf download drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored \
  --local-dir ~/models/dsv4-flash-dspark-abliterated
MODELDIR=~/models/dsv4-flash-dspark-abliterated bash scripts/dsv4-nvfp4-1m-serve.sh 1
MODELDIR=~/models/dsv4-flash-dspark-abliterated bash scripts/dsv4-nvfp4-1m-serve.sh 0
```

Details: [RESULTS.md](RESULTS.md) · [docs/STATUS.md](docs/STATUS.md)

---

## v1.1 alpha — Mida / Brikie / Hermes-friendly

**HF (separate repo, same gate):**  
https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-v1.1-alpha-Mida-Brikie  

| | |
|---|---|
| Context | **1,048,576** |
| Standing concurrency | **C=4** |
| C1 pure | **~50 tok/s** |
| C4 agg | **~113 tok/s** |
| Refusal | **32/32 = 100%** · 0 garble |
| Weights | L0–9 stock · L10–42 ablit · **MTP stock** · λ=3.5 |

```bash
hf download drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-v1.1-alpha-Mida-Brikie \
  --local-dir ~/models/dsv4-flash-dspark-abliterated-mida

# C=4 standing (1M)
MODELDIR=~/models/dsv4-flash-dspark-abliterated-mida \
  CTX=1048576 SEQS=4 UTIL=0.85 bash scripts/serve-v1.1-alpha.sh 1
MODELDIR=~/models/dsv4-flash-dspark-abliterated-mida \
  CTX=1048576 SEQS=4 UTIL=0.85 bash scripts/serve-v1.1-alpha.sh 0
```

Raw numbers: [results/v1.1-alpha/](results/v1.1-alpha/) · [docs/STATUS_V1.1_ALPHA.md](docs/STATUS_V1.1_ALPHA.md)

---

## Image (both)

```bash
docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
docker tag  ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10 \
  vllm-dspark-runtime:dspark-nvfp4-stage-c
```

Edit `MASTER` / `IF` / `HCA` in the serve scripts for your fabric. **gpu-clear both ranks before relaunch.** Rank1 first, then rank0.

---

## Hermes / agent ops (still required)

Even v1.1 can echo a fat skills catalog if Hermes says “MUST skill_view”. Use **on-demand** skills, names-only index, `max_tokens: 8192`, `tool_use_enforcement: false`.  
See [docs/HERMES_SPILL_FIX.md](docs/HERMES_SPILL_FIX.md).

---

## Repo layout

```
RELEASES.md      # v1.0 vs v1.1 chooser (start here)
RESPONSIBLE_USE.md
RESULTS.md       # v1.0 performance write-up
results/         # v1.0 raw JSON
results/v1.1-alpha/  # v1.1 probes + concurrency + ABLIT_META
docs/
scripts/         # serve + project_wob (supports --no-mtp for v1.1)
charts/
```

## Base model

[deepseek-ai/DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)

## Disclaimer

Research releases. Outputs may include content stock models refuse. Do not deploy without your own safety layer. No liability for misuse.
