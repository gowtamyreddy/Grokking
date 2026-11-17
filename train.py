"""Grokking reproduction: modular addition with a tiny transformer.

Paper: Power et al., "Grokking: Generalization Beyond Overfitting on
Small Algorithmic Datasets" (arXiv:2201.02177). A small transformer
trained on a+b mod p memorizes the training set fast, then, long
after train accuracy hits 100%, validation accuracy suddenly jumps
to ~100%. That delayed jump is grokking.

Setup (the paper's, shrunk to laptop scale): one decoder layer,
d_model 128, 4 heads, p = 97, half of all pairs for training, the
rest held out, full-batch AdamW with lr 1e-3 and weight decay 1.0.
Weight decay is the ingredient that makes it grok; without it the
model memorizes and never generalizes.

Run:
  .venv/bin/python train.py [--steps 20000] [--p 97] [--train-frac 0.5]
  .venv/bin/python train.py --smoke   (300 steps, checks it runs)

Logs one JSON line per 100 steps to runs/metrics.jsonl, and prints
the step where validation first passes 99% ("grok steps").
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


class TinyTransformer(nn.Module):
    """One decoder layer, built from parts, so every piece is visible."""

    def __init__(self, vocab, d_model=128, n_head=4, d_ff=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(16, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        seq = x.shape[1]
        pos = torch.arange(seq, device=x.device)
        h = self.tok(x) + self.pos(pos)
        mask = torch.triu(torch.ones(seq, seq, device=x.device,
                                     dtype=torch.bool), diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=mask)
        h = self.ln1(h + a)
        h = self.ln2(h + self.mlp(h))
        return self.head(h)


def make_data(p, train_frac, seed=0):
    """All (a, b) pairs; the target is (a + b) mod p.

    Sequence is [a, b, =], one extra token for the equals sign, and
    the prediction is read off the last position.
    """
    g = torch.Generator().manual_seed(seed)
    pairs = [(a, b) for a in range(p) for b in range(p)]
    random.Random(seed).shuffle(pairs)
    n_train = int(len(pairs) * train_frac)
    train_pairs, val_pairs = pairs[:n_train], pairs[n_train:]

    def to_tensors(pairs):
        x = torch.tensor(
            [[a, b, p] for a, b in pairs], dtype=torch.long
        )
        y = torch.tensor([(a + b) % p for a, b in pairs], dtype=torch.long)
        return x, y

    return to_tensors(train_pairs), to_tensors(val_pairs)


@torch.no_grad()
def accuracy(model, x, y):
    logits = model(x)[:, -1, :]
    return (logits.argmax(dim=-1) == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="runs/metrics.jsonl")
    parser.add_argument("--threads", type=int, default=2,
                        help="torch threads (small on shared CPUs)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    if args.smoke:
        args.steps = 300

    torch.manual_seed(args.seed)
    (x_train, y_train), (x_val, y_val) = make_data(
        args.p, args.train_frac, seed=args.seed
    )
    vocab = args.p + 1  # residues 0..p-1 plus the "=" token
    model = TinyTransformer(vocab, d_model=args.d_model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.98),
                            weight_decay=args.weight_decay)
    print(f"p={args.p} train={len(x_train)} val={len(x_val)} "
          f"params={sum(p.numel() for p in model.parameters())}")
    print(f"chance accuracy: {100 / args.p:.2f}%")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grok_step = None
    start = time.time()
    with open(out_path, "w") as log:
        for step in range(1, args.steps + 1):
            logits = model(x_train)[:, -1, :]
            loss = F.cross_entropy(logits, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == 1:
                train_acc = accuracy(model, x_train, y_train)
                val_acc = accuracy(model, x_val, y_val)
                row = {
                    "step": step,
                    "train_loss": round(loss.item(), 4),
                    "train_acc": round(train_acc, 4),
                    "val_acc": round(val_acc, 4),
                    "seconds": round(time.time() - start, 1),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"step {step:6d} loss {row['train_loss']:.4f} "
                      f"train {train_acc:.3f} val {val_acc:.3f} "
                      f"({row['seconds']}s)")
                if grok_step is None and val_acc >= 0.99:
                    grok_step = step
                    print(f"GROKKED at step {step}: val acc {val_acc:.3f}")
                    break
    print(f"done in {time.time() - start:.1f}s; "
          f"grok step: {grok_step if grok_step else 'not reached'}; "
          f"metrics in {out_path}")


if __name__ == "__main__":
    main()
