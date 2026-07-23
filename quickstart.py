#!/usr/bin/env python3
"""
CLAD quickstart — runs entirely from the bundled data (no downloads, CPU is fine).

Step 1: load the pre-built Parser Bank (compiled offline by the foundry) and
        parse the 11 bundled Loghub-2k systems with deterministic function
        matching only — no Drain, no LLM at runtime — then score PA/GA/TA
        against the corrected ground truth.
Step 2: load the shipped real-time classifier checkpoint and classify the
        bundled labeled BGL event-stream sample (sliding windows of 80 events).

Usage:  python quickstart.py [bank]   (default: loghub-2k)
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
BANK = sys.argv[1] if len(sys.argv) > 1 else "loghub-2k"


def step1_parser():
    print("=" * 70)
    print(f"STEP 1: CLAD-Parser on Loghub-2k (Parser Bank: {BANK})")
    print("=" * 70)
    subprocess.run([sys.executable, str(ROOT / "parser/eval/run_parser_2k.py"), BANK], check=True)
    subprocess.run([sys.executable, str(ROOT / "parser/eval/score_parser.py"), "CLAD-Parser"], check=True)


# Same architecture as the training/testing notebooks (classifier/, realtime/)
class EventSequenceModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dropout=.3), num_layers=2)
        self.attn = nn.Linear(embed_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(.3), nn.Linear(hidden_dim, 2))

    def forward(self, x):
        b, s = x.size()
        x = self.embed(x) + self.pos_encoder(
            torch.arange(s, device=x.device).unsqueeze(0).expand(b, s))
        x = self.transformer(x.permute(1, 0, 2))
        w = torch.softmax(self.attn(x), dim=0)
        return self.fc((x * w).sum(dim=0))


def step2_classifier():
    print()
    print("=" * 70)
    print("STEP 2: CLAD-Classifier on bundled labeled BGL sample")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = ROOT / "realtime/BGL/models"
    meta = json.load(open(mdir / "training_metadata_BGL.json"))
    vocab = json.load(open(mdir / "event_vocab_BGL.json"))

    model = EventSequenceModel(meta["vocab_size"], meta["embed_dim"], meta["hidden_dim"])
    model.load_state_dict(torch.load(
        mdir / "TableVIII_bgl_realtime_classifier.pth", weights_only=True, map_location=device))
    model.to(device).eval()

    df = pd.read_csv(ROOT / "data/samples/labeled_logs_BGL_sample.csv")
    df["EventIdx"] = df["EventId"].map(vocab).fillna(len(vocab) - 1).astype(int)
    L, B = meta["sequence_length"], meta["block_size"]

    seqs, labels = [], []
    for start in range(0, len(df) - B, B):
        blk = df.iloc[start:start + B]
        ev, lb = blk["EventIdx"].tolist(), blk["Label"].tolist()
        for i in range(0, B - L, L):
            seqs.append(ev[i:i + L])
            labels.append(1 if 1 in lb[i:i + L] else 0)

    X, y = torch.tensor(seqs), torch.tensor(labels)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            preds.append(model(X[i:i + 256].to(device)).argmax(1).cpu())
    p = torch.cat(preds)

    tp = ((p == 1) & (y == 1)).sum().item()
    fp = ((p == 1) & (y == 0)).sum().item()
    fn = ((p == 0) & (y == 1)).sum().item()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"device={device}  sequences={len(y)}  anomaly_fraction={y.float().mean():.3f}")
    print(f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print("\nNote: the bundled sample overlaps the checkpoint's training data;")
    print("this is a smoke test of the pipeline, not a held-out evaluation.")
    print("For the full benchmarks see realtime/BGL and realtime/Thunderbird.")


if __name__ == "__main__":
    step1_parser()
    step2_classifier()
    print("\nQuickstart finished.")
