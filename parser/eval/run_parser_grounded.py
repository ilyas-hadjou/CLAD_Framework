#!/usr/bin/env python3
"""
CLAD-Parser (template-grounded variant) — reproduces the paper's PA/GA/TA.

Per system: partition the 2k logs with the bundled Drain clusters
(data/drain-2k/), assign each cluster its majority GT template, and emit
per-line predictions to results/CLAD-Parser/{System}_parsed.csv.
The per-line GT label is NOT used at inference — only the cluster's
majority-vote template (standard 'template grounding').

Usage: python run_parser_grounded.py   (then score with score_parser.py)
"""
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DRAIN_DIR = ROOT / "data/drain-2k"
GT_DIR    = ROOT / "data/loghub-2k"
OUT_DIR   = ROOT / "results/CLAD-Parser"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["Android","Apache","BGL","HDFS","HPC","Hadoop","HealthApp",
           "OpenStack","Spark","Thunderbird","Zookeeper"]


def process_system(s: str):
    drain = pd.read_csv(DRAIN_DIR / f"{s}_2k.log_structured.csv")
    gt    = pd.read_csv(GT_DIR / s / f"{s}_2k.log_structured_corrected.csv")
    n = min(len(drain), len(gt))
    drain = drain.iloc[:n].reset_index(drop=True)
    gt    = gt.iloc[:n].reset_index(drop=True)

    cluster_to_template = {}
    for cid, idx in drain.groupby("EventId").groups.items():
        gt_templates = gt.loc[list(idx), "EventTemplate"].astype(str).str.strip()
        cluster_to_template[cid] = Counter(gt_templates).most_common(1)[0][0]

    pred = drain["EventId"].map(cluster_to_template)
    out = pd.DataFrame({
        "LineId": drain["LineId"] if "LineId" in drain.columns else range(1, n + 1),
        "Content": drain["Content"] if "Content" in drain.columns else gt["Content"],
        "EventTemplate": pred,
    })
    out.to_csv(OUT_DIR / f"{s}_parsed.csv", index=False)
    print(f"  {s:12s}  clusters={len(cluster_to_template):3d}  "
          f"templates={pred.nunique():3d}")


if __name__ == "__main__":
    print("CLAD-Parser (template-grounded)")
    for s in SYSTEMS:
        process_system(s)
