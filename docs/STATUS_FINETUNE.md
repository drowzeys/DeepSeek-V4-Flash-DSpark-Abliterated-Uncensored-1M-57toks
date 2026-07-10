# Fine-tune: 100% bypass + Hermes-clean (2026-07-10)

## Recipe (weights)
- Base: stock DSV4-Flash-DSpark
- Abliterate `attn.wo_b` on **layers 10–42 + mtp** only
- Early layers **0–9 remain stock** (protocol / chat / tools)
- Method: SRA rank-1, **λ=3.5**
- Path: `~/models/dsv4-flash-dspark-abliterated` (1M serve on `.3`/`.4`)

## Hermes-side (required for "perfect Hermes")
1. Skills prompt patched to **on-demand** (not "mandatory / MUST load") in  
   `~/.hermes/hermes-agent/agent/prompt_builder.py`
2. `model.max_tokens: 8192`, `temperature: 0`
3. `tool_use_enforcement: false`
4. Strong `environment_hint` against catalog paste
5. **Restart Hermes + `/new`** after these changes

## Live stack
```
max_model_len=1048576
kv=nvfp4_ds_mla ~2.39M
container=dsv4_60
API=http://10.100.10.3:8000/v1
```

## How to re-apply weights
```bash
cd ~/dsv4-ablit
python3 project_wob.py --src ~/models/dsv4-flash-dspark \
  --dst ~/models/dsv4-flash-dspark-abliterated \
  --direction work/refusal_direction_r1.pt \
  --lambda-attn 3.5 --min-layer 10 --max-layer 42 --n-directions 1
```
