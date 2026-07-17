#!/usr/bin/env python3
"""
train.py - REFERENCE training script: a character-level MLP from scratch.

This is the reusable char-LM baseline shipped with the template. Swap in your
own data/model as needed, but KEEP the export contract (parameter names/shapes
below) so scripts/export_model.py and web/inference.js keep working.

This is Karpathy's "makemore" part 2 (a Bengio 2003-style neural bigram/n-gram):
embed a fixed window of `block_size` previous characters, run them through one
tanh hidden layer, and project to a distribution over the next character.

The model is deliberately tiny (~a few thousand params) so the whole forward
pass can be re-implemented in a few lines of browser JavaScript for the live
demo (see ../web and ../scripts/export_model.py). The parameter names and shapes
defined here are the CONTRACT the exporter and the JS backend both rely on:

    C  : [vocab, emb]                 character embedding table
    W1 : [block_size * emb, hidden]   b1 : [hidden]
    W2 : [hidden, vocab]              b2 : [vocab]

Everything is seeded, so a fresh `python train/train.py` reproduces the exact
checkpoint that ships in this repo.

Usage:
    python train/train.py                 # train + write train/checkpoint.pt
    python train/train.py --steps 40000   # override any hyperparameter
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


def load_names(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build_vocab(names: list[str]) -> list[str]:
    """vocab[0] is '.', the start/end boundary token; then the sorted letters.
    This ordering is the contract the JS backend assumes (boundary == index 0)."""
    chars = sorted({c for name in names for c in name})
    return ["."] + chars


def build_dataset(names, stoi, block_size):
    """Turn each name into (context, next-char) training pairs.
    A name 'ali' with block_size 3 yields:
        ... -> a,  ..a -> l,  .al -> i,  ali -> .   (the final '.' teaches ending)
    """
    X, Y = [], []
    for name in names:
        context = [0] * block_size  # start padded with the boundary token
        for ch in name + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # slide the window
    return torch.tensor(X), torch.tensor(Y)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Turkish-name char-level MLP.")
    ap.add_argument("--names", default=str(ROOT / "data" / "names.txt"))
    ap.add_argument("--out", default=str(ROOT / "train" / "checkpoint.pt"))
    ap.add_argument("--block-size", type=int, default=3)
    ap.add_argument("--embedding-dim", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # --- reproducibility: seed every source of randomness we use ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)  # used for BOTH init and batching

    names = load_names(Path(args.names))
    vocab = build_vocab(names)
    stoi = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    print(f"{len(names)} names | vocab={V} | block_size={args.block_size}")

    # 80/10/10 split by name, so we can watch for overfitting honestly.
    random.shuffle(names)
    n1, n2 = int(0.8 * len(names)), int(0.9 * len(names))
    Xtr, Ytr = build_dataset(names[:n1], stoi, args.block_size)
    Xdev, Ydev = build_dataset(names[n1:n2], stoi, args.block_size)
    Xte, Yte = build_dataset(names[n2:], stoi, args.block_size)
    print(f"train={tuple(Xtr.shape)} dev={tuple(Xdev.shape)} test={tuple(Xte.shape)}")

    # --- parameters (names/shapes are the export contract) ---
    emb, hidden, B = args.embedding_dim, args.hidden, args.block_size
    C = torch.randn((V, emb), generator=g)
    W1 = torch.randn((B * emb, hidden), generator=g) * (5 / 3) / (B * emb) ** 0.5  # tanh gain / fan-in
    b1 = torch.zeros(hidden)
    W2 = torch.randn((hidden, V), generator=g) * 0.01  # small so initial logits are ~uniform
    b2 = torch.zeros(V)
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True
    print(f"parameters: {sum(p.nelement() for p in params)}")

    def forward(X):
        emb_ = C[X].view(X.shape[0], -1)          # [batch, block*emb]
        h = torch.tanh(emb_ @ W1 + b1)            # [batch, hidden]
        return h @ W2 + b2                        # logits [batch, vocab]

    @torch.no_grad()
    def split_loss(X, Y):
        return F.cross_entropy(forward(X), Y).item()

    # --- training loop: minibatch SGD with step-decay + weight decay ---
    # 561 names is a tiny corpus, so the model will happily memorise. We fight
    # that two ways: a small L2 penalty (weight decay), and early stopping -
    # we snapshot whichever weights had the LOWEST dev loss, not the last ones.
    best_dev, best_sd = float("inf"), None
    wd = args.weight_decay
    for i in range(args.steps):
        ix = torch.randint(0, Xtr.shape[0], (args.batch_size,), generator=g)  # seeded batches
        loss = F.cross_entropy(forward(Xtr[ix]), Ytr[ix])
        for p in params:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < args.steps * 0.6 else 0.01
        for p in params:
            p.data += -lr * (p.grad + wd * p.data)  # gradient step + L2 pull toward 0
        if i % 2000 == 0 or i == args.steps - 1:
            dev_now = split_loss(Xdev, Ydev)
            if dev_now < best_dev:  # early stopping: keep the best-generalising snapshot
                best_dev = dev_now
                best_sd = {k: v.detach().clone() for k, v in
                           zip("C W1 b1 W2 b2".split(), params)}
            print(f"  step {i:6d}/{args.steps}  batch {loss.item():.4f}  dev {dev_now:.4f}")

    # restore the best-dev snapshot before reporting / saving
    for p, k in zip(params, "C W1 b1 W2 b2".split()):
        p.data = best_sd[k]
    tr, dev = split_loss(Xtr, Ytr), split_loss(Xdev, Ydev)
    print(f"best-dev loss  train {tr:.4f}  dev {dev:.4f}  test {split_loss(Xte, Yte):.4f}")

    # --- save a checkpoint so the trained weights are never lost again ---
    sd = {"C": C.detach(), "W1": W1.detach(), "b1": b1.detach(),
          "W2": W2.detach(), "b2": b2.detach()}
    torch.save(
        {"model": sd,
         "config": {"vocab": vocab, "block_size": B, "embedding_dim": emb, "hidden": hidden},
         "loss": {"train": tr, "dev": dev}},
        args.out,
    )
    print(f"saved checkpoint -> {args.out}")

    # --- sample a few names as a sanity check ---
    print("\nsamples:")
    gs = torch.Generator().manual_seed(args.seed + 1)
    for _ in range(15):
        out, context = [], [0] * B
        while True:
            logits = forward(torch.tensor([context]))
            ix = torch.multinomial(F.softmax(logits, dim=1), 1, generator=gs).item()
            if ix == 0:
                break
            out.append(vocab[ix])
            context = context[1:] + [ix]
        print("  " + "".join(out))


if __name__ == "__main__":
    main()
