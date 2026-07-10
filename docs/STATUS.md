# Local abliteration status — DSV4-Flash-DSpark

**Date:** 2026-07-09/10  
**Goal:** ≥90% refusal bypass (ideally ~100%)  
**Result:** **100% bypass on 32-prompt battery** · residual refusal **0%** · harmless coherence **OK**

## Checkpoint

| | |
|---|---|
| Path | `~/models/dsv4-flash-dspark-abliterated` (on `.3` and `.4`) |
| Base | `~/models/dsv4-flash-dspark` (unchanged) |
| Method | Direct FP8 `attn.wo_b` + `mtp.wo_b` projection (mHC-resistant path) |
| Directions | **Rank-1** broad refusal dir, **SRA-cleaned** (r=8 capability atoms) |
| λ | **3.5** (stronger than lovesenko’s 2.5) |
| Edit footprint | 46 tensors (43 decoder + 3 MTP), Δrel Frobenius ~0.05–0.06 |
| Meta | `ABLIT_META.json` in checkpoint dir |

## Why not stock obliteratus / lovesenko drop-in

- Stock obliteratus LoRA-style fails on mHC (edits renormalized away).
- lovesenko residual ~7% on broad set was not enough for our goal.
- k=6 multi-dir SVD **without** SRA (first attempt) hit ~62% bypass but **destroyed coherence** (CN/EN garble loops) — abandoned.

## Serve (current)

```bash
# rank1 then rank0 (.4 / .3)
bash ~/dsv4-ablit/serve-abliterated.sh 1
bash ~/dsv4-ablit/serve-abliterated.sh 0   # on .3
# API http://10.100.10.3:8000/v1
```

Container: `dsv4_ablit_srv` · image `vllm-dspark-runtime:dspark-nvfp4-stage-c` · `kv=nvfp4_ds_mla` · DSpark k=5 · TP=2.

## Probe files

- `work/refusal_probe_r1_l35.json` — **final 100% bypass**
- `work/refusal_probe_ablit.json` — failed multi-dir attempt (coherence damage)
- `work/refusal_direction_r1.pt` / `refusal_direction_sra.pt`

## Pipeline scripts

`serve-capture.sh` · `run_capture.py` · `compute_direction.py` · `project_wob.py` · `serve-abliterated.sh` · `patches/`
