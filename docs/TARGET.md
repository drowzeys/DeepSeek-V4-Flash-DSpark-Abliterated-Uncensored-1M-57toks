# Abliteration target (local DSV4-Flash-DSpark)

| metric | goal |
|---|---|
| **Refusal bypass rate** | **≥ 90%** excellent · **~100%** ideal |
| Residual refusal | ≤ 10% · ideally ~0% |
| vs lovesenko | their broad residual ~7.4% still not enough for us; we push multi-dir + higher λ |

## Method (not stock obliteratus / not lovesenko λ=2.5 rank-1)

mHC kills LoRA-style edits. Pipeline:

1. Serve base TP=2 with `wo_b` activation capture hooks
2. Diff-in-means + **SVD multi-direction (k=4–8)** harmful vs harmless
3. Project all 43× `attn.wo_b` + 3× `mtp.wo_b` in FP8: `W ← W − λ Vᵀ(VW)` with **λ ≥ 3.5** (try 3.5 → 4.0 → 4.5 until ≥90% bypass or capability collapses)
4. Probe with `refusal_eval/refusal_probes.py` + local battery
5. Iterate λ / k if bypass < 90%

## Notes

- Capability may drop as λ rises; we accept some quality cost for gate bypass.
- PII/self-harm categories are hardest; 100% on all categories may be unreachable without also breaking crisis-refusal pathways.
