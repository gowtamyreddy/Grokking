"""Plot the grokking curve from runs/metrics.jsonl.

Reads the JSON lines train.py writes and draws train vs validation
accuracy against step, the picture the paper is famous for: train
accuracy flat at 100% while validation sits near chance, then
suddenly jumps to ~100%.

Run: .venv/bin/python plot.py [--in runs/metrics.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="runs/metrics.jsonl")
    parser.add_argument("--out", default="runs/curve.png")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(ROOT / args.inp)]
    steps = [r["step"] for r in rows]
    train = [r["train_acc"] for r in rows]
    val = [r["val_acc"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train, label="train accuracy")
    ax.plot(steps, val, label="validation accuracy")
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Grokking: a+b mod 97, one-layer transformer")
    ax.legend()
    ax.grid(alpha=0.3)
    out = ROOT / args.out
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    grok = next((r["step"] for r in rows if r["val_acc"] >= 0.99), None)
    print(f"wrote {out}; grok step: {grok if grok else 'not reached'}")


if __name__ == "__main__":
    main()
