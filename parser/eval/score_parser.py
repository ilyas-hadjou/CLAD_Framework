#!/usr/bin/env python3
"""
Score the per-line CSVs under results/{Parser}/{System}_parsed.csv against
the corrected Loghub-2.0 GT, write
  results/_full/{Parser}/metrics_per_system.csv
in the same format as build_full_metrics_table.py.

Usage: python scripts/score_local_parser.py CLAD-Parser [LogPrompt ...]
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT/"data/loghub-2k"
OUT_BASE = ROOT/"results/_full"

SYS = ["Android","Apache","BGL","HDFS","HPC","Hadoop","HealthApp","OpenStack","Spark","Thunderbird","Zookeeper"]

def _norm(s): return s.astype(str).str.strip()

def PA(dp,dt): return float((_norm(dp["EventTemplate"])==_norm(dt["EventTemplate"])).mean())
def GA(dp,dt):
    pred = dp.groupby(_norm(dp["EventTemplate"])).indices
    truth= dt.groupby(_norm(dt["EventTemplate"])).indices
    n=len(dp); correct=0
    for _,idx in truth.items():
        s=set(idx)
        for _,pidx in pred.items():
            if set(pidx)==s: correct+=len(s); break
    return correct/n if n else 0.0
def TA(dp,dt):
    t=set(_norm(dt["EventTemplate"]).unique())
    p=set(_norm(dp["EventTemplate"]).unique())
    return len(p&t)/len(t) if t else 0.0

def score_parser(parser):
    src = ROOT/"results"/parser
    if not src.exists():
        print(f"[ERR] {src} missing"); return
    out_dir = OUT_BASE/parser; out_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for s in SYS:
        pp = src/f"{s}_parsed.csv"
        gt = pd.read_csv(GT_DIR/s/f"{s}_2k.log_structured_corrected.csv")
        if not pp.exists():
            rows.append({"system":s,"PA":None,"GA":None,"TA":None}); continue
        try:
            dp = pd.read_csv(pp)
        except Exception as e:
            print(f"  [{s}] read err: {e}")
            rows.append({"system":s,"PA":None,"GA":None,"TA":None}); continue
        if "EventTemplate" not in dp.columns:
            rows.append({"system":s,"PA":None,"GA":None,"TA":None}); continue
        n=min(len(dp),len(gt))
        dp=dp.iloc[:n].reset_index(drop=True); dt=gt.iloc[:n].reset_index(drop=True)
        rows.append({"system":s,
                     "PA":round(PA(dp,dt),4),
                     "GA":round(GA(dp,dt),4),
                     "TA":round(TA(dp,dt),4)})
    df=pd.DataFrame(rows)
    df.to_csv(out_dir/"metrics_per_system.csv", index=False)
    avg={"system":"Average",
         "PA":round(df["PA"].mean(skipna=True),4),
         "GA":round(df["GA"].mean(skipna=True),4),
         "TA":round(df["TA"].mean(skipna=True),4)}
    pd.concat([df,pd.DataFrame([avg])], ignore_index=True).to_csv(
        out_dir/"metrics_with_avg.csv", index=False)
    print(f"\n=== {parser} ===")
    print(df.to_string(index=False))
    print(f"AVG  PA={avg['PA']}  GA={avg['GA']}  TA={avg['TA']}")

if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: score_local_parser.py PARSER [PARSER ...]"); sys.exit(1)
    for p in sys.argv[1:]:
        score_parser(p)
