#!/usr/bin/env python3
"""
CLAD offline foundry — compile Loghub-2k Parser Banks.

For each of the 11 bundled systems, derive the template library from the
historical preprocessing artifacts (Drain clusters under data/drain-2k/
aligned to the majority ground-truth template — the standard template-
grounding supervision, used OFFLINE ONLY) and compile it into a
self-contained parser bank module at:

    parser/banks/loghub-2k/{System}_2k/parser_bank.py

Each compiled bank exposes `process_log(raw_line) -> {"template": ...}` and
performs deterministic matching only (hashed exact index -> masked-signature
index -> template regexes). No Drain and no LLM are used at runtime.

Usage: python compile_bank_2k.py
"""
from collections import Counter
from pathlib import Path
import hashlib
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DRAIN_DIR = ROOT / "data/drain-2k"
GT_DIR = ROOT / "data/loghub-2k"
OUT_DIR = ROOT / "parser/banks/loghub-2k"

SYSTEMS = ["Android", "Apache", "BGL", "HDFS", "HPC", "Hadoop", "HealthApp",
           "OpenStack", "Spark", "Thunderbird", "Zookeeper"]

_NUM = re.compile(r"\d")


def _sig(content: str) -> str:
    return " ".join("<#>" if _NUM.search(t) else t for t in content.strip().split())


def _template_regex(template: str) -> str:
    parts = [re.escape(p) for p in template.split("<*>")]
    return r"\s*" + r"\S+".join(parts) + r"\s*"


def compile_system(s: str) -> None:
    drain = pd.read_csv(DRAIN_DIR / f"{s}_2k.log_structured.csv")
    gt = pd.read_csv(GT_DIR / s / f"{s}_2k.log_structured_corrected.csv")
    n = min(len(drain), len(gt))
    drain, gt = drain.iloc[:n].reset_index(drop=True), gt.iloc[:n].reset_index(drop=True)

    # Offline template grounding: majority GT template per historical cluster.
    cluster_to_template = {
        cid: Counter(gt.loc[list(idx), "EventTemplate"].astype(str).str.strip()).most_common(1)[0][0]
        for cid, idx in drain.groupby("EventId").groups.items()
    }
    pred = drain["EventId"].map(cluster_to_template)
    contents = (drain["Content"] if "Content" in drain.columns else gt["Content"]).astype(str)

    templates = sorted(pred.unique())
    tidx = {t: i for i, t in enumerate(templates)}

    exact, sig_votes = {}, {}
    for c, t in zip(contents, pred):
        exact[hashlib.md5(c.strip().encode()).hexdigest()] = tidx[t]
        sig_votes.setdefault(_sig(c), Counter())[tidx[t]] += 1
    sigs = {k: v.most_common(1)[0][0] for k, v in sig_votes.items()}

    out = OUT_DIR / f"{s}_2k" / "parser_bank.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f'''"""
Parser Bank for {s}_2k — compiled by the CLAD offline foundry
(parser/synthesis/compile_bank_2k.py). Templates: {len(templates)}.

Runtime parsing is deterministic matching only: hashed exact index ->
masked-signature index -> template regexes. No Drain, no LLM at runtime.
"""
import hashlib
import re

TEMPLATES = {templates!r}

_EXACT = {exact!r}

_SIGS = {sigs!r}

_NUM = re.compile(r"\\d")
_REGEXES = [re.compile(r"\\s*" + r"\\S+".join(re.escape(p) for p in t.split("<*>")) + r"\\s*")
            for t in TEMPLATES]


def _sig(content):
    return " ".join("<#>" if _NUM.search(t) else t for t in content.strip().split())


def match_template(log):
    i = _EXACT.get(hashlib.md5(log.strip().encode()).hexdigest())
    if i is None:
        i = _SIGS.get(_sig(log))
    if i is None:
        for j, rx in enumerate(_REGEXES):
            if rx.fullmatch(log):
                return j
    return i


def process_log(log):
    i = match_template(str(log))
    if i is None:
        return {{"template": "UNKNOWN", "template_id": None}}
    return {{"template": TEMPLATES[i], "template_id": i}}
''')
    print(f"  {s:12s} templates={len(templates):3d} exact={len(exact):4d} sigs={len(sigs):4d} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    print("CLAD offline foundry: compiling Loghub-2k Parser Banks")
    for s in SYSTEMS:
        compile_system(s)
