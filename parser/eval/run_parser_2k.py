#!/usr/bin/env python3
"""
Run the CLAD-Parser over the 11 Loghub-2k systems and save per-line
predictions to results/CLAD-Parser/{System}_parsed.csv.

Each system has a parser bank at:
  parser/banks/{MODEL}/{System}_2k/parser_bank.py
which exposes `process_log(raw_line) -> {"template": str, ...}`.

Ground-truth `Content` column is read from:
  data/loghub-2k/{System}/{System}_2k.log_structured_corrected.csv

Usage: python run_parser_2k.py [qwen2.5-7b|qwen3-30b]   (default: qwen3-30b)
"""
import importlib.util
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BANK_MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-30b"
BANK_DIR = ROOT / "parser/banks" / BANK_MODEL
GT_DIR   = ROOT / "data/loghub-2k"
OUT_DIR  = ROOT / "results/CLAD-Parser"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["Android","Apache","BGL","HDFS","HPC","Hadoop","HealthApp",
           "OpenStack","Spark","Thunderbird","Zookeeper"]

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def predict_template(mod, raw: str) -> str:
    try:
        out = mod.process_log(raw)
        if isinstance(out, dict):
            t = out.get("template", "UNKNOWN")
            return str(t) if t is not None else "UNKNOWN"
        return str(out)
    except Exception:
        return "UNKNOWN"

def main():
    summary = []
    for sysname in SYSTEMS:
        bank = BANK_DIR / f"{sysname}_2k" / "parser_bank.py"
        gt_csv = GT_DIR / sysname / f"{sysname}_2k.log_structured_corrected.csv"
        if not bank.exists():
            print(f"[SKIP] {sysname}: no parser bank at {bank}")
            continue
        if not gt_csv.exists():
            print(f"[SKIP] {sysname}: no GT at {gt_csv}")
            continue

        mod = load_module(bank, f"clad_{sysname.lower()}")
        df_gt = pd.read_csv(gt_csv)
        contents = df_gt["Content"].astype(str).tolist()

        preds = [predict_template(mod, c) for c in contents]
        out_df = pd.DataFrame({
            "LineId": df_gt["LineId"] if "LineId" in df_gt.columns else range(1, len(preds)+1),
            "Content": contents,
            "EventTemplate": preds,
        })
        out_path = OUT_DIR / f"{sysname}_parsed.csv"
        out_df.to_csv(out_path, index=False)

        unknown = sum(1 for p in preds if p == "UNKNOWN")
        n_tmpl  = len(set(preds))
        print(f"{sysname:12s}  n={len(preds)}  templates={n_tmpl:3d}  unknown={unknown:4d}  -> {out_path.relative_to(ROOT)}")
        summary.append({"system": sysname, "n": len(preds),
                        "n_templates": n_tmpl, "unknown": unknown})

    pd.DataFrame(summary).to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nWrote {OUT_DIR/'_summary.csv'}")

if __name__ == "__main__":
    main()
