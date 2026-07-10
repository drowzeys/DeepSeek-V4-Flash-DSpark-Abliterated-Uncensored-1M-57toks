# DeepSeek-V4-Flash-DSpark — Abliterated (Uncensored) · 1M · ~57 tok/s

**Weights (Hugging Face):**  
https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored

Local abliterated build of **DeepSeek-V4-Flash-DSpark** for **2× NVIDIA DGX Spark (GB10)** with:

| | |
|---|---|
| Context | **1,048,576** (`nvfp4_ds_mla` KV) |
| C1 pure | **~57 tok/s** (code decode, TP=2 stage-c, DSpark k=5) |
| Refusal bypass | **~100%** on 32-prompt battery + hard probe |
| Hermes | works with **on-demand** skills prompt (not “mandatory MUST load”) |

> Research / local use only. Removes most stock safety refusals.

## What’s in the weights

Hybrid **layer-range** abliteration (mHC-resistant; LoRA does not work on this family):

- **Stock** `attn.wo_b` · layers **0–9** (chat / tools / protocol)
- **Abliterated** `attn.wo_b` · layers **10–42** + **MTP** draft heads
- SRA-cleaned **rank-1** refusal direction · **λ = 3.5**
- FP8 dequant → project → requant (same DSpark shard layout)

Base: [deepseek-ai/DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)

## Quick start (2× DGX Spark)

```bash
# 0. Image (GB10 stage-c / B12X / graphs)
docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
docker tag  ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10 \
  vllm-dspark-runtime:dspark-nvfp4-stage-c

# 1. Weights on BOTH nodes (~157 GB)
hf download drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored \
  --local-dir ~/models/dsv4-flash-dspark-abliterated

# 2. Edit MASTER/IF/HCA in scripts/dsv4-nvfp4-1m-serve.sh
# 3. Rank1 first, then rank0 (API :8000)
MODELDIR=~/models/dsv4-flash-dspark-abliterated \
  bash scripts/dsv4-nvfp4-1m-serve.sh 1
MODELDIR=~/models/dsv4-flash-dspark-abliterated \
  bash scripts/dsv4-nvfp4-1m-serve.sh 0
```

## Rebuild abliteration (optional)

```bash
# After capturing refusal directions (see scripts/prompts.py + compute_direction.py)
python3 scripts/project_wob.py \
  --src ~/models/dsv4-flash-dspark \
  --dst ~/models/dsv4-flash-dspark-abliterated \
  --direction work/refusal_direction_r1.pt \
  --lambda-attn 3.5 --min-layer 10 --max-layer 42 --n-directions 1
```

## Hermes agent

Abliterated models will **echo the skills catalog** if Hermes still uses:

> “Skills (mandatory) … MUST load with skill_view … Err on the side of loading”

Use the **on-demand** skills rules (patch or equivalent):

- Greetings / simple Q → plain short text, **no** `skill_view`
- Never paste the skills index into the reply
- Load skills only for concrete multi-step tasks

Also recommended: `model.max_tokens: 8192`, `temperature: 0`, `tool_use_enforcement: false`.

See [docs/HERMES_SPILL_FIX.md](docs/HERMES_SPILL_FIX.md) and [docs/STATUS_FINETUNE.md](docs/STATUS_FINETUNE.md).

## Repo layout

```
scripts/
  dsv4-nvfp4-1m-serve.sh   # 1M TP=2 serve (stage-c)
  serve-abliterated.sh     # simpler ablit serve helper
  project_wob.py           # FP8 wo_b abliteration
  hybrid_overlay.py        # restore stock early layers
  compute_direction.py     # direction extraction
  prompts.py               # capture / eval prompts
docs/
  STATUS_FINETUNE.md
  ABLIT_META.json
  eval_tune_final.json
```

## Measured (publish cluster)

- Topology: TP=2 · nodes `10.100.10.3` + `.4` · 200G RoCE  
- Image: `vllm-dspark-runtime:dspark-nvfp4-stage-c`  
- KV pool @ 1M: ~2.39M tokens  
- C1 pure: ~57 tok/s mean class on code prompts  

## Disclaimer

Research release. Outputs may include content stock models refuse. Do not deploy without your own safety layer. No liability for misuse.

## Links

- Weights: https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored  
- Stage-c image (optional): `ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10`  
- Base model: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark  
