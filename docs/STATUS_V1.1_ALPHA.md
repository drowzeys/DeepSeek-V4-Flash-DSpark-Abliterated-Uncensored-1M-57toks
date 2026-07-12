# DSV4-Flash-DSpark — balance recipe (bypass + Mida/Brikie/Hermes)

**Date:** 2026-07-12  
**Goal:** ≥95% refusal bypass (prefer 100%) while minimizing Hermes skill-spill / Brikie–Mida protocol damage.  
**Result:** **WINNING — 100% bypass · 0 garble · clean greetings · MTP stock**

## Standing balance recipe (ship this)

| | |
|---|---|
| Base | Pristine stock restored from **.3** `~/models/dsv4-flash-dspark` |
| Path | `~/models/dsv4-flash-dspark-abliterated-mida` |
| Direction | `work/refusal_direction_r1.pt` rank-1 SRA |
| Layers | **L10–42** `attn.wo_b` only |
| Stock kept | **L0–9** + **all MTP** (`--no-mtp`) |
| λ | **3.5** |
| Edits | 33 tensors · mean Δrel ≈ 0.057 |

```bash
# after pristine stock is on this node:
cd ~/dsv4-ablit
rm -rf ~/models/dsv4-flash-dspark-abliterated-mida
python3 project_wob.py \
  --src ~/models/dsv4-flash-dspark \
  --dst ~/models/dsv4-flash-dspark-abliterated-mida \
  --direction work/refusal_direction_r1.pt \
  --lambda-attn 3.5 --min-layer 10 --max-layer 42 \
  --n-directions 1 --no-mtp
```

`project_wob.py` writes via temp+replace (no hardlink write-through). **Never** `save_file` onto a hardlinked stock shard.

## Live test (.1 / .2)

| | |
|---|---|
| Nodes | `10.100.10.1` rank0 · `10.100.10.2` rank1 |
| API | `http://10.100.10.1:8000/v1` |
| Model | `deepseek-v4-flash-dspark` |
| Image | `vllm-dspark-runtime:dspark-nvfp4-stage-c` |
| Launch | `~/dsv4-ablit/serve-abliterated-c12.sh {0,1}` |
| UTIL | **0.85** · DSpark k=5 · kv=`nvfp4_ds_mla` · ctx=262144 |

### Probe results (Recipe A — balance)

| Metric | Score |
|---|---|
| Refusal battery | **21/21 = 100%** bypass · 0 refuse · 0 garble |
| Coherence | **5/5** |
| Hermes greet (fat skills index in system) | **3/3** — no catalog dump on `hello` / `hi there` |
| Tool protocol | **4/4** |
| C1 count decode | **~63.4 tok/s** |

Probe: `probe_balance.py` → `work/probe_balance_A.json`

**Note on “what can you do?”:** may still *mention* skills by name (e.g. Mermaid) without dumping the full `- mermaid: …` catalog. That is much milder than the reported initial skill leak; names-only Hermes index + on-demand skills prompt remain recommended ops.

## Why this is “best of both worlds”

| Prior failure | Fix in this recipe |
|---|---|
| User report: initial Hermes skill catalog spill | L0–9 stock + MTP stock; no multi-dir hot ablit |
| User report: Brikie/Mida repeated-use garble/spill | Avoid λ≥5 / multi-dir; keep early protocol layers |
| Need high bypass | Same L10–42 window + λ=3.5 as standing (minus MTP edit) |
| Stock write-through corruption on .4 | Restored full stock from **.3** (Jun 28 pristine) |

Compared to published standing (L10–42 λ3.5 **+ MTP**): this keeps **MTP stock** (GLM / Mida lesson) while matching **100%** on the 21-prompt hard battery.

## Ladder (this campaign)

| Recipe | Bypass | Coh | Hermes greet | Tool | Notes |
|---|---|---|---|---|---|
| L20–42 λ3.0 no-mtp (clean early) | ~58% | 3/3 | — | 4/4 | Too mild |
| L10–42 λ3.0 no-mtp (dirty stock era) | ~25% | 3/3 | — | 4/4 | Invalid — stock pollution |
| **L10–42 λ3.5 no-mtp (restored stock)** | **100%** | **5/5** | **3/3** | **4/4** | **WINNING** |
| Prior standing L10–42 λ3.5 +MTP | 100% (32) | OK | ops-dependent | ops | More MTP touch |

## Ops still required (do not skip)

Weights alone do not fix fat Hermes skill catalogs in production:

1. **Names-only** skills index in `prompt_builder.py`  
2. On-demand skills (not “MUST skill_view”)  
3. `max_tokens: 8192`, `tool_use_enforcement: false`  
4. Spill hooks allowlisted (`skill_spill_guard` / `skill_spill_pre`)  
5. After spill: `lcm_scrub.py --apply` + `/new`

## Stock recovery (if .4 polluted again)

```bash
# .3 has pristine Jun-28 stock
STAGE=/var/tmp/dsv4-stock-from-3
rsync -aH keyspark@10.100.10.3:~/models/dsv4-flash-dspark/ $STAGE/
# break hardlinks then install
for f in $STAGE/model-*.safetensors; do
  base=$(basename "$f")
  rm -f ~/models/dsv4-flash-dspark/$base
  cp -al "$f" ~/models/dsv4-flash-dspark/$base || cp -a "$f" ~/models/dsv4-flash-dspark/$base
done
```

## Artifacts

- Weights: `~/models/dsv4-flash-dspark-abliterated-mida` (fanout on .1/.2)  
- Project log: `work/project_balance_l10_lam35_nomtp.log`  
- Probe: `work/probe_balance_A.json` · `work/probe_balance_A_summary.json`  
- Scripts: `project_wob.py`, `serve-abliterated-c12.sh`, `probe_balance.py`
