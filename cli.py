#!/usr/bin/env python3
"""
CLAD command-line interface.

Lightweight wrapper around the CLAD pipeline used by quickstart.py:
  * parse    -- run a compiled Parser Bank over raw log lines (file or literal string)
  * classify -- run the real-time anomaly classifier over an EventId stream
                (CSV with an `EventId` column, optionally a `Label` column)
  * both     -- parse first, then classify

Examples:
  python cli.py --input "generating core.128" --mode parse
  python cli.py --input data/samples/labeled_logs_BGL_sample.csv --mode classify
  python cli.py --input mylogs.log --mode both --output results.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "realtime/BGL/models/TableVIII_bgl_realtime_classifier.pth"
MODEL_META_DIR = ROOT / "realtime/BGL/models"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_input(inp: str):
    """Return (lines, dataframe). dataframe is set only for CSV inputs."""
    p = Path(inp)
    if p.exists():
        if p.suffix.lower() == ".csv":
            import pandas as pd
            return None, pd.read_csv(p)
        return p.read_text(errors="replace").splitlines(), None
    # Not an existing path: treat as a literal log line / text string.
    return inp.splitlines(), None


def run_parse(args, lines, df):
    """Parse raw log lines with a compiled parser bank (reuses parser/eval helpers)."""
    parser_eval = _load_module(ROOT / "parser/eval/run_parser_2k.py", "clad_parser_eval")

    if lines is None:
        if df is not None and "Content" in df.columns:
            lines = df["Content"].astype(str).tolist()
        else:
            print("[parse] SKIP: CSV input has no 'Content' column with raw log text.")
            return None

    bank_path = ROOT / "parser/banks" / args.bank / f"{args.system}_2k" / "parser_bank.py"
    if not bank_path.exists():
        sys.exit(f"[parse] ERROR: no parser bank at {bank_path}")

    bank = parser_eval.load_module(bank_path, f"clad_bank_{args.system.lower()}")
    results = [{"line": ln, "template": parser_eval.predict_template(bank, ln)} for ln in lines]

    n_templates = len({r["template"] for r in results})
    unknown = sum(1 for r in results if r["template"] == "UNKNOWN")
    print(f"[parse] bank={args.bank}/{args.system}_2k  lines={len(results)}  "
          f"templates={n_templates}  unknown={unknown}")
    for r in results[:5]:
        print(f"        {r['line'][:60]!r} -> {r['template'][:60]!r}")
    return {"bank": f"{args.bank}/{args.system}_2k", "num_lines": len(results),
            "num_templates": n_templates, "unknown": unknown, "results": results}


def run_classify(args, df, parse_result=None):
    """Classify an event stream with the real-time checkpoint (arch from quickstart.py)."""
    import torch
    from quickstart import EventSequenceModel

    meta = json.load(open(MODEL_META_DIR / "training_metadata_BGL.json"))
    vocab = json.load(open(MODEL_META_DIR / "event_vocab_BGL.json"))
    oov = len(vocab) - 1

    if df is not None and "EventId" in df.columns:
        events = [vocab.get(str(e), oov) for e in df["EventId"]]
        labels = df["Label"].tolist() if "Label" in df.columns else None
    elif parse_result is not None:
        # Raw text path: map parsed templates to (mostly OOV) vocab indices.
        events = [vocab.get(r["template"], oov) for r in parse_result["results"]]
        labels = None
    else:
        print("[classify] SKIP: need a CSV with an 'EventId' column "
              "(or run --mode both on raw text).")
        return None

    L = meta["sequence_length"]
    if len(events) < L:
        print(f"[classify] SKIP: need at least {L} events, got {len(events)}.")
        return None

    seqs, ys = [], []
    for i in range(0, len(events) - L + 1, L):
        seqs.append(events[i:i + L])
        if labels is not None:
            ys.append(1 if 1 in labels[i:i + L] else 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.exists():
        sys.exit(f"[classify] ERROR: checkpoint not found: {ckpt}")

    model = EventSequenceModel(meta["vocab_size"], meta["embed_dim"], meta["hidden_dim"])
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.to(device).eval()

    X = torch.tensor(seqs)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            preds.append(model(X[i:i + 256].to(device)).argmax(1).cpu())
    p = torch.cat(preds)

    out = {"checkpoint": str(ckpt), "device": str(device),
           "num_windows": len(p), "num_anomalous": int((p == 1).sum()),
           "predictions": p.tolist()}
    print(f"[classify] device={device}  windows={out['num_windows']}  "
          f"anomalous={out['num_anomalous']}")

    if labels is not None:
        y = torch.tensor(ys)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out["metrics"] = {"precision": prec, "recall": rec, "f1": f1}
        print(f"[classify] Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="CLAD CLI: contextual log parsing "
                                             "and real-time anomaly detection.")
    ap.add_argument("--input", required=True,
                    help="Path to an input log file / CSV, or a literal log text string.")
    ap.add_argument("--mode", choices=["parse", "classify", "both"], default="both",
                    help="Operation mode (default: both).")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="Path to a pre-trained classifier checkpoint "
                         f"(default: {DEFAULT_CHECKPOINT.relative_to(ROOT)}).")
    ap.add_argument("--output", default=None,
                    help="Optional path to save results as JSON.")
    ap.add_argument("--system", default="BGL",
                    help="Parser bank system, e.g. BGL, HDFS, Apache (default: BGL).")
    ap.add_argument("--bank", default="qwen2.5-7b", choices=["qwen2.5-7b", "qwen3-30b"],
                    help="Parser bank model family (default: qwen2.5-7b).")
    args = ap.parse_args()

    lines, df = _read_input(args.input)
    results = {"input": args.input, "mode": args.mode}

    parse_result = None
    if args.mode in ("parse", "both"):
        parse_result = run_parse(args, lines, df)
        if parse_result is not None:
            results["parse"] = parse_result
    if args.mode in ("classify", "both"):
        classify_result = run_classify(args, df, parse_result)
        if classify_result is not None:
            results["classify"] = classify_result

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
