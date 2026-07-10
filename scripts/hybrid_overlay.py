#!/usr/bin/env python3
"""Overlay stock functional wo_b onto abliterated checkpoint (in-place).

Keeps abliteration on late layers (refusal-heavy), restores stock early layers
so Hermes system/tool protocols work better.

Default: stock layers 0..keep_stock_until-1, abliterated layers keep_stock_until..42 + mtp.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", type=Path, default=Path.home() / "models/dsv4-flash-dspark")
    ap.add_argument("--ablit", type=Path, default=Path.home() / "models/dsv4-flash-dspark-abliterated")
    ap.add_argument(
        "--keep-stock-until",
        type=int,
        default=20,
        help="Layers [0, N) use stock wo_b; layers [N, 43) + mtp stay abliterated",
    )
    ap.add_argument("--stock-mtp", action="store_true", help="Also restore stock mtp.wo_b")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    idx = json.loads((args.ablit / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]

    # which wo_b to restore from stock
    restore = []
    for name, shard in wm.items():
        if not name.endswith("attn.wo_b.weight"):
            continue
        if name.startswith("mtp."):
            if args.stock_mtp:
                restore.append(name)
            continue
        # layers.N.attn.wo_b.weight
        try:
            lid = int(name.split(".")[1])
        except Exception:
            continue
        if lid < args.keep_stock_until:
            restore.append(name)

    # also their scales
    targets = set()
    for n in restore:
        targets.add(n)
        targets.add(n.replace(".weight", ".scale"))

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in targets:
        by_shard[wm[name]].append(name)

    print(f"restore {len(restore)} wo_b weights from stock (layers < {args.keep_stock_until}"
          f"{' + mtp' if args.stock_mtp else ''})")
    print(f"shards to rewrite: {len(by_shard)}")

    stats = []
    for shard, names in sorted(by_shard.items()):
        sp = args.stock / shard
        apath = args.ablit / shard
        print(f"patch {shard}: {sorted(names)}", flush=True)
        tensors = {}
        with safe_open(str(apath), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        with safe_open(str(sp), framework="pt") as f:
            for name in names:
                stock_t = f.get_tensor(name)
                ablit_t = tensors[name]
                if stock_t.shape != ablit_t.shape:
                    raise SystemExit(f"shape mismatch {name}: stock {stock_t.shape} ablit {ablit_t.shape}")
                # relative fro before
                if name.endswith(".weight"):
                    # dequant rough compare only on float cast of fp8
                    d = (stock_t.float() - ablit_t.float()).norm() / (stock_t.float().norm().clamp_min(1e-8))
                    stats.append({"tensor": name, "pre_overlay_rel": float(d), "source": "stock"})
                tensors[name] = stock_t
        if args.dry_run:
            print("  dry-run skip write")
            continue
        # write temp then replace
        tmp = apath.with_suffix(".safetensors.tmp")
        out = {k: v.contiguous() for k, v in tensors.items()}
        save_file(out, str(tmp))
        tmp.replace(apath)
        print(f"  wrote {apath}", flush=True)

    meta_path = args.ablit / "ABLIT_META.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta["hybrid_overlay"] = {
        "keep_stock_until": args.keep_stock_until,
        "stock_mtp": args.stock_mtp,
        "restored_wo_b": restore,
        "n_restored": len(restore),
        "note": "Early-layer stock wo_b for Hermes/tool protocol; late-layer abliteration for refusal.",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print("DONE hybrid overlay", meta["hybrid_overlay"]["n_restored"], "tensors")
    if stats:
        print("sample pre-overlay deltas:", stats[:3])


if __name__ == "__main__":
    main()
