#!/usr/bin/env python3
"""Apply rank-1 refusal projection to FP8 attn.wo_b (+ mtp.wo_b) shards.

W <- W - λ * v * (v^T W)   with v in R^{4096} (output dim)
Preserves row structure in the 4096-d output space (lovesenko-style).

FP8: weight float8_e4m3fn, scale float8_e8m0fnu with 128x128 blocks.
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """weight [N,K] e4m3, scale [N/B, K/B] e8m0 → bf16."""
    w = weight.to(torch.float32)
    # e8m0 stores power-of-two scales; cast via float works on modern torch
    s = scale.to(torch.float32)
    n, k = w.shape
    assert s.shape == ((n + block - 1) // block, (k + block - 1) // block), (
        f"scale {tuple(s.shape)} vs weight {tuple(w.shape)}"
    )
    # expand scales
    s_exp = s.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:k]
    # fix if last dims truncated
    if s_exp.shape != w.shape:
        s_full = torch.ones_like(w)
        nb, kb = s.shape
        for i in range(nb):
            for j in range(kb):
                s_full[i * block : min((i + 1) * block, n), j * block : min((j + 1) * block, k)] = s[i, j]
        s_exp = s_full
    return (w * s_exp).to(torch.bfloat16)


def quant_fp8_block(w_bf16: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N,K] → e4m3 weight + e8m0 block scales (amax-based)."""
    w = w_bf16.to(torch.float32)
    n, k = w.shape
    nb = (n + block - 1) // block
    kb = (k + block - 1) // block
    scales = torch.empty((nb, kb), dtype=torch.float32)
    for i in range(nb):
        for j in range(kb):
            tile = w[i * block : min((i + 1) * block, n), j * block : min((j + 1) * block, k)]
            amax = tile.abs().amax().clamp_min(1e-12)
            # e4m3 max ~448
            scale = amax / 448.0
            # snap to power-of-two for e8m0 friendliness
            exp = torch.ceil(torch.log2(scale.clamp_min(1e-30)))
            scale = torch.pow(torch.tensor(2.0), exp)
            scales[i, j] = scale
            w[i * block : min((i + 1) * block, n), j * block : min((j + 1) * block, k)] = tile / scale
    w_fp8 = w.clamp(-448, 448).to(torch.float8_e4m3fn)
    # float8_e8m0fnu from float powers of two
    try:
        s_fp8 = scales.to(torch.float8_e8m0fnu)
    except Exception:
        # fallback: store as float8_e4m3 scale isn't ideal; use float32 saved as e8m0 via view trick
        s_fp8 = scales.to(torch.float8_e8m0fnu)
    return w_fp8, s_fp8


def project(W: torch.Tensor, V: torch.Tensor, lam: float) -> torch.Tensor:
    """W [out,in], V [k,out] orthonormal rows (or [out] / [out,k]) → multi-dir proj.

    W <- W - λ * V^T (V W)   with V as [k, out] unit rows.
    lovesenko used k=1 λ=2.5; we default stronger k>1 and higher λ.
    """
    if V.dim() == 1:
        V = V.view(1, -1)
    V = V.to(W.dtype)
    # ensure [k, out]
    if V.shape[1] != W.shape[0] and V.shape[0] == W.shape[0]:
        V = V.T
    # re-orthonormalize
    Q = []
    for i in range(V.shape[0]):
        v = V[i]
        for q in Q:
            v = v - torch.dot(v, q) * q
        n = v.norm().clamp_min(1e-8)
        Q.append(v / n)
    V = torch.stack(Q, 0)  # [k, out]
    # VW = V @ W → [k, in]
    VW = V @ W
    return W - lam * (V.transpose(0, 1) @ VW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/home/keyspark/models/dsv4-flash-dspark"))
    ap.add_argument("--dst", type=Path, default=Path("/home/keyspark/models/dsv4-flash-dspark-abliterated"))
    ap.add_argument("--direction", type=Path, default=Path("/home/keyspark/dsv4-ablit/work/refusal_direction.pt"))
    ap.add_argument("--lambda-attn", type=float, default=3.5, help="lovesenko used 2.5; we push harder")
    ap.add_argument("--min-layer", type=int, default=0, help="first decoder layer to abliterate (inclusive)")
    ap.add_argument("--max-layer", type=int, default=42, help="last decoder layer to abliterate (inclusive)")
    ap.add_argument("--no-mtp", action="store_true", help="do not edit mtp.wo_b")
    ap.add_argument("--n-directions", type=int, default=0, help="0=all in file; else first k")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = torch.load(args.direction, map_location="cpu", weights_only=False)
    broad = payload["broad"].float()
    deep = payload.get("deep", broad).float()
    per_layer = {int(k): v.float() for k, v in payload.get("per_layer", {}).items()}
    if "directions" in payload and payload["directions"] is not None:
        dirs = payload["directions"].float()
        if dirs.dim() == 1:
            dirs = dirs.unsqueeze(0)
    else:
        dirs = broad.unsqueeze(0)
    if args.n_directions and args.n_directions > 0:
        dirs = dirs[: args.n_directions]
    print(f"using {dirs.shape[0]} directions, lambda={args.lambda_attn}, layers=[{args.min_layer},{args.max_layer}] mtp={not args.no_mtp}")

    idx = json.loads((args.src / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]

    # tensors to edit (layer-range hybrid)
    targets = []
    for name, shard in weight_map.items():
        if not name.endswith("attn.wo_b.weight"):
            continue
        if name.startswith("mtp."):
            if args.no_mtp:
                continue
            targets.append((name, shard))
            continue
        try:
            lid = int(name.split(".")[1])
        except Exception:
            continue
        if args.min_layer <= lid <= args.max_layer:
            targets.append((name, shard))
    print(f"targets: {len(targets)} wo_b.weight tensors (layers {args.min_layer}-{args.max_layer})")

    # group by shard
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    edit_names = {n for n, _ in targets}
    # also need scales stay with weights - we recompute scales
    scale_of = {n.replace(".weight", ".scale"): n for n in edit_names if n.endswith(".weight")}

    args.dst.mkdir(parents=True, exist_ok=True)
    # copy non-weight files
    for p in args.src.iterdir():
        if p.name.startswith("model-") and p.name.endswith(".safetensors"):
            continue
        if p.name == "model.safetensors.index.json":
            continue
        dest = args.dst / p.name
        if p.is_dir():
            if dest.exists():
                continue
            shutil.copytree(p, dest, dirs_exist_ok=True)
        else:
            if not dest.exists():
                shutil.copy2(p, dest)

    stats = []
    shards_to_write = set()
    for name, shard in targets:
        shards_to_write.add(shard)

    for shard_name, keys in sorted(by_shard.items()):
        src_path = args.src / shard_name
        dst_path = args.dst / shard_name
        needs_edit = any(k in edit_names for k in keys)
        if not needs_edit:
            # hardlink unchanged shards
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            try:
                os.link(src_path, dst_path)
            except OSError:
                shutil.copy2(src_path, dst_path)
            print(f"link {shard_name}")
            continue

        print(f"edit {shard_name} ...", flush=True)
        # Always base on STOCK so early/hybrid layers stay pristine even if dst was dirty
        tensors = {}
        with safe_open(str(src_path), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        for name in list(tensors.keys()):
            if name not in edit_names:
                continue
            scale_name = name.replace(".weight", ".scale")
            W_fp8 = tensors[name]
            S = tensors[scale_name]
            # which direction?
            # multi-dir basis: global SVD dirs (stronger). mtp uses same basis
            # (deepest-layer component already in SVD pool).
            V = dirs
            W = dequant_fp8_block(W_fp8, S)
            assert W.shape[0] == V.shape[-1] or W.shape[0] == V.shape[0], (
                f"{name} out={W.shape[0]} V={tuple(V.shape)}"
            )
            Wp = project(W, V, args.lambda_attn)
            # row-magnitude preserve (optional light): scale rows back
            row_n0 = W.norm(dim=1).clamp_min(1e-8)
            row_n1 = Wp.norm(dim=1).clamp_min(1e-8)
            Wp = Wp * (row_n0 / row_n1).unsqueeze(1)
            delta = (Wp.float() - W.float()).norm() / W.float().norm()
            W_new, S_new = quant_fp8_block(Wp)
            tensors[name] = W_new
            tensors[scale_name] = S_new
            stats.append({"tensor": name, "rel_fro": float(delta), "shape": list(W.shape)})
            print(f"  {name} Δrel={delta:.4f}", flush=True)

        if args.dry_run:
            print("dry-run skip write", shard_name)
            continue
        # safetensors needs contiguous cpu tensors
        out = {k: (t.contiguous() if torch.is_tensor(t) else t) for k, t in tensors.items()}
        # CRITICAL: if dst is a hardlink to stock (or anything else), writing would
        # corrupt the shared inode. Always write via temp then replace.
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        save_file(out, str(tmp_path))
        tmp_path.replace(dst_path)
        print(f"  wrote {dst_path}", flush=True)

    # index unchanged
    shutil.copy2(args.src / "model.safetensors.index.json", args.dst / "model.safetensors.index.json")
    meta_path = args.dst / "ABLIT_META.json"
    meta_path.write_text(json.dumps({
        "method": "layer-range-wo_b-projection",
        "lambda_attn": args.lambda_attn,
        "min_layer": args.min_layer,
        "max_layer": args.max_layer,
        "edit_mtp": not args.no_mtp,
        "n_directions": int(dirs.shape[0]),
        "direction": str(args.direction),
        "n_edited": len(stats),
        "stats": stats,
        "note": "mHC-resistant family; LoRA abliteration ineffective. Direct FP8 wo_b edit.",
    }, indent=2))
    print("DONE", args.dst, "edited", len(stats))


if __name__ == "__main__":
    main()
