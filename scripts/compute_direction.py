#!/usr/bin/env python3
"""Diff-in-means + multi-direction SVD refusal basis from captured activations.

Stronger than lovesenko rank-1: keep top-k SVD directions of the
harmful-vs-harmless residual matrix for aggressive projection.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def load_stack(tag_dir: Path, n_layers: int = 43) -> dict[int, torch.Tensor]:
    out = {}
    for lid in range(n_layers):
        p = tag_dir / f"layer_{lid:02d}.pt"
        if not p.exists():
            continue
        raw = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 1:
                raw = raw.unsqueeze(0)
            vecs = raw.float()
        else:
            vecs = torch.stack([v.float().view(-1) for v in raw], dim=0)
        out[lid] = vecs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/home/keyspark/dsv4-ablit/work"))
    ap.add_argument("--out", type=Path, default=Path("/home/keyspark/dsv4-ablit/work/refusal_direction.pt"))
    ap.add_argument("--n-layers", type=int, default=43)
    ap.add_argument("--n-directions", type=int, default=4, help="SVD rank (lovesenko used 1)")
    args = ap.parse_args()

    harm = load_stack(args.work / "harmful", args.n_layers)
    safe = load_stack(args.work / "harmless", args.n_layers)
    common = sorted(set(harm) & set(safe))
    if not common:
        raise SystemExit(f"no common layers under {args.work}")

    # Per-layer mean diff units
    layer_units = []
    seps = {}
    per_layer = {}
    centered_rows = []
    for lid in common:
        mh = harm[lid].mean(0)
        ms = safe[lid].mean(0)
        d = mh - ms
        seps[lid] = float(d.norm().item())
        u = d / d.norm().clamp_min(1e-8)
        per_layer[lid] = u
        layer_units.append(u)
        # also center each harmful sample vs harmless mean for SVD pool
        for i in range(harm[lid].shape[0]):
            centered_rows.append(harm[lid][i] - ms)

    broad = torch.stack(layer_units, 0).mean(0)
    broad = broad / broad.norm().clamp_min(1e-8)

    # Multi-direction: SVD on [n_samples*layers, H] residual matrix
    M = torch.stack(centered_rows, 0)  # [N, H]
    # also append per-layer mean diffs
    M = torch.cat([M, torch.stack(layer_units, 0)], 0)
    # economy SVD
    try:
        # M ≈ U S Vh with Vh [k, H]
        _, S, Vh = torch.linalg.svd(M, full_matrices=False)
        k = min(args.n_directions, Vh.shape[0])
        dirs = Vh[:k].clone()
        # Gram-Schmidt vs first (should already be ortho from SVD)
        for i in range(k):
            dirs[i] = dirs[i] / dirs[i].norm().clamp_min(1e-8)
        singular = S[:k].tolist()
    except Exception as e:
        print("SVD failed, falling back to broad only:", e)
        dirs = broad.unsqueeze(0)
        singular = [float(seps[max(seps, key=seps.get)])]
        k = 1

    deep_lid = max(common)
    deep = per_layer[deep_lid]

    payload = {
        "broad": broad,
        "deep": deep,
        "directions": dirs,  # [k, H]
        "singular_values": singular,
        "per_layer": {str(k_): v for k_, v in per_layer.items()},
        "separation": seps,
        "n_harmful": {str(k_): int(harm[k_].shape[0]) for k_ in common},
        "n_harmless": {str(k_): int(safe[k_].shape[0]) for k_ in common},
        "layers": common,
        "n_directions": int(dirs.shape[0]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(json.dumps({
        "layers": common,
        "hidden": int(broad.numel()),
        "n_directions": int(dirs.shape[0]),
        "singular": singular,
        "sep_top5": sorted(seps.items(), key=lambda x: -x[1])[:5],
        "n_harm_mean": sum(payload["n_harmful"].values()) / len(common),
        "out": str(args.out),
    }, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
