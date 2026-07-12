# Restore: DSV4F Ablit v1.1-alpha Mida/Brikie · C=4 · 1M

**Snapshot id:** `dsv4f-ablit-v1.1-mida-c4-1m`  
**Path:** `~/checkpoints/dsv4f-ablit-v1.1-mida-c4-1m`  
**Created:** see `meta/IDENTITY.txt`

This freezes the **known-good state** after:

- Milder ablit (L10–42 λ=3.5, **MTP stock**)
- 32/32 refusal bypass
- Standing serve **C=4 · 1M · UTIL 0.85 · DSpark k=5**
- Hermes primary → `http://10.100.10.1:8000/v1` · `deepseek-v4-flash-dspark`
- Published HF + GitHub

---

## What is frozen here

| Piece | Location |
|---|---|
| Identity / URLs / image digest | `meta/IDENTITY.txt` |
| Ablit recipe | `meta/ABLIT_META.json` |
| Refusal direction | `meta/refusal_direction_r1.pt` |
| Weight fingerprints (L0/9/10/20/42 + mtp) | `meta/WEIGHT_FINGERPRINTS.json` |
| Shard hardlink map | `meta/SHARD_INVENTORY.json` |
| Serve script (C=4 default) | `scripts/serve-v1.1-alpha.sh` |
| Project script (hardlink-safe) | `scripts/project_wob.py` |
| Hermes config (DSV4F primary) | `hermes/config.yaml` |
| Probes / concurrency / hermes stress | `results/` |
| Live serve log excerpt | `meta/live_serve_log_excerpt.txt` |

**Weights themselves** are **not** re-copied (157G). They live at:

- Local: `~/models/dsv4-flash-dspark-abliterated-mida`
- HF: `drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-v1.1-alpha-Mida-Brikie` (gated)

Stock (pristine source of truth on fleet):

- Preferred: `keyspark@10.100.10.3:~/models/dsv4-flash-dspark`
- Local: `~/models/dsv4-flash-dspark` (restored from .3 when building this release)

---

## Quick restore (same fleet, weights still on disk)

```bash
SNAP=~/checkpoints/dsv4f-ablit-v1.1-mida-c4-1m
bash "$SNAP/restore.sh"
```

What `restore.sh` does:

1. Verifies weight fingerprints vs `WEIGHT_FINGERPRINTS.json` (optional re-download from HF if fail)
2. Restores Hermes config from snapshot + restarts gateway
3. Installs serve/project scripts into `~/dsv4-ablit/`
4. Relaunches TP=2 on **.1 (rank0) + .2 (rank1)** with **C=4 · 1M · UTIL 0.85**

---

## Manual restore

### 1. Weights

```bash
# if local tree missing or fingerprints fail:
hf download drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-v1.1-alpha-Mida-Brikie \
  --local-dir ~/models/dsv4-flash-dspark-abliterated-mida

# optional: re-project from pristine stock (.3) instead of download
# rsync stock from .3 first (break hardlinks on install — see STATUS_MIDA.md)
python3 ~/checkpoints/dsv4f-ablit-v1.1-mida-c4-1m/scripts/project_wob.py \
  --src ~/models/dsv4-flash-dspark \
  --dst ~/models/dsv4-flash-dspark-abliterated-mida \
  --direction ~/checkpoints/dsv4f-ablit-v1.1-mida-c4-1m/meta/refusal_direction_r1.pt \
  --lambda-attn 3.5 --min-layer 10 --max-layer 42 --n-directions 1 --no-mtp
```

### 2. Image

```bash
docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
docker tag ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10 \
  vllm-dspark-runtime:dspark-nvfp4-stage-c
# expect digest: sha256:76532c4cc261afe7a7cad1d9731cd5123d0e14219c9a1d35a0ef6163fe67c5d4
```

### 3. Serve (C=4 · 1M)

```bash
# fanout weights + script to .1/.2 if needed
# rank1 then rank0
ssh keyspark@10.100.10.2 'CTX=1048576 SEQS=4 UTIL=0.85 bash ~/dsv4-ablit/serve-abliterated-c12.sh 1'
ssh keyspark@10.100.10.1 'CTX=1048576 SEQS=4 UTIL=0.85 bash ~/dsv4-ablit/serve-abliterated-c12.sh 0'
# API: http://10.100.10.1:8000/v1
```

### 4. Hermes

```bash
cp -a ~/checkpoints/dsv4f-ablit-v1.1-mida-c4-1m/hermes/config.yaml ~/.hermes/config.yaml
# restart gateway (kill by PID from: ps -eo pid,cmd | awk '/hermes_cli.main gateway/')
# then /new on open chats
```

### 5. Verify

```bash
# fingerprints
python3 ~/checkpoints/dsv4f-ablit-v1.1-mida-c4-1m/verify_weights.py

# refusal 32/32
# (use historical suite prompts + live API)

# hermes default hits DSV4F:
# agent.log should show base_url=http://10.100.10.1:8000/v1 model=deepseek-v4-flash-dspark
```

---

## Published anchors (durable off-box)

| Artifact | URL |
|---|---|
| GitHub (docs + scripts) | https://github.com/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks @ `2be3332` |
| HF v1.1 Mida/Brikie (gated) | https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-v1.1-alpha-Mida-Brikie |
| HF v1.0 full ablit (gated) | https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored |
| RELEASES chooser | GitHub `RELEASES.md` |

---

## Do not confuse with v1.0

| | v1.0 | **v1.1 (this snapshot)** |
|---|---|---|
| MTP | abliterated | **stock** |
| C1 | ~57 tok/s | **~50 tok/s** |
| Standing C | often 12 | **4** |
| Local tree | `…/dsv4-flash-dspark-abliterated` | **`…/dsv4-flash-dspark-abliterated-mida`** |
