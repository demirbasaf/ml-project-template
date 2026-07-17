#!/usr/bin/env python3
"""
export_model.py - turn a trained checkpoint into the browser artifact.

Reads train/checkpoint.pt (produced by train/train.py) and writes the two files
the web demo and the personal website both consume, following the shared
ARTIFACT CONVENTION:

    model/
      ├── meta.json     # vocab, block_size, hyperparams, sampling defaults
      └── weights.json  # every parameter as { shape, data(flat, row-major) }

The parameter names/shapes below are the contract with the browser forward pass
(web/inference.js and the website's js-backend). For a model this small (~8k
params) JSON is the right format: no runtime, no WASM, just numbers.

Usage:
    python scripts/export_model.py                    # uses train/checkpoint.pt
    python scripts/export_model.py --checkpoint x.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Export checkpoint.pt -> model/ artifact.")
    ap.add_argument("--checkpoint", default=str(ROOT / "train" / "checkpoint.pt"))
    ap.add_argument("--out", default=str(ROOT / "model"))
    ap.add_argument("--name", default="turkish-makemore")
    ap.add_argument("--title", default="Turkish name generator")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd, cfg = ckpt["model"], ckpt["config"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # weights.json: flat, row-major data arrays keyed by parameter name.
    params = {}
    for pname, tensor in sd.items():
        arr = tensor.detach().cpu().float().numpy()
        params[pname] = {"shape": list(arr.shape), "data": arr.reshape(-1).tolist()}
    n_params = sum(len(p["data"]) for p in params.values())
    (out / "weights.json").write_text(json.dumps({"format": "flat", "params": params}))

    # meta.json: everything the browser needs to render + run the model, no weights.
    meta = {
        "name": args.name,
        "title": args.title,
        "description": "Character-level MLP that invents Turkish first names, trained from scratch.",
        "backend": "js",
        "arch": "mlp",
        "vocab": cfg["vocab"],
        "block_size": cfg["block_size"],
        "embedding_dim": cfg["embedding_dim"],
        "hidden_size": cfg["hidden"],
        "sampling": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k or None,
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"wrote {out}/weights.json ({n_params} params) + meta.json")
    print(f"  vocab={len(cfg['vocab'])} block_size={cfg['block_size']} "
          f"emb={cfg['embedding_dim']} hidden={cfg['hidden']}")
    if "loss" in ckpt:
        print(f"  checkpoint loss: train {ckpt['loss']['train']:.4f} dev {ckpt['loss']['dev']:.4f}")


if __name__ == "__main__":
    main()
