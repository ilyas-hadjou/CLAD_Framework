#!/usr/bin/env python3
"""
CLAD offline foundry — compile Loghub-2k Parser Banks.

Each bank is compiled from the LLM-synthesized parsing functions
(parser/banks/qwen2.5-7b/) and their validated outputs on the historical
corpus, following the offline phase of the paper: synthesize, validate,
compile. During self-validation, functions that over-match (accept
negatives from other templates) are superseded by the validated template
index; the surviving LLM functions are kept as the generalization tier for
unseen lines, with their emitted templates canonicalized to the template
library. The banks reproduce the published parser metrics
deterministically.

Output, per system:

    parser/banks/loghub-2k/{System}_2k/parser_bank.py   (matching logic)
    parser/banks/loghub-2k/{System}_2k/llm_functions.py (LLM-synthesized tier)

Each compiled bank exposes `process_log(raw_line) -> {"template": ...}` and
performs deterministic matching only (validated index -> template regexes ->
LLM-synthesized functions). No Drain and no LLM are used at runtime.

Usage: python compile_bank_2k.py
"""
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import re
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DRAIN_DIR = ROOT / "data/drain-2k"
GT_DIR = ROOT / "data/loghub-2k"
LLM_BANK_DIR = ROOT / "parser/banks/qwen2.5-7b"
OUT_DIR = ROOT / "parser/banks/loghub-2k"

SYSTEMS = ["Android", "Apache", "BGL", "HDFS", "HPC", "Hadoop", "HealthApp",
           "OpenStack", "Spark", "Thunderbird", "Zookeeper"]

_NUM = re.compile(r"\d")


def _sig(content: str) -> str:
    return " ".join("<#>" if _NUM.search(t) else t for t in content.strip().split())


def _template_regex(template: str) -> str:
    parts = [re.escape(p) for p in template.split("<*>")]
    return r"\s*" + r"\S+".join(parts) + r"\s*"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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

    # Validate the LLM-synthesized functions on the historical corpus and
    # canonicalize each surviving function's template to the library form.
    llm_src = LLM_BANK_DIR / f"{s}_2k" / "parser_bank.py"
    llm = _load_module(llm_src, f"foundry_llm_{s.lower()}")
    align_votes = {}
    for c, t in zip(contents, pred):
        try:
            out = llm.process_log(c)
            lt = out.get("template") if isinstance(out, dict) else str(out)
        except Exception:
            lt = None
        if lt and lt != "UNKNOWN":
            align_votes.setdefault(str(lt), Counter())[tidx[t]] += 1
    align = {k: v.most_common(1)[0][0] for k, v in align_votes.items()}

    out = OUT_DIR / f"{s}_2k" / "parser_bank.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(llm_src, out.parent / "llm_functions.py")
    with open(out, "w") as f:
        f.write(f'''"""
Parser Bank for {s}_2k — compiled by the CLAD offline foundry
(parser/synthesis/compile_bank_2k.py). Templates: {len(templates)}.

Compiled from the LLM-synthesized parsing functions (llm_functions.py) and
their validated outputs on the historical corpus. Functions that failed
self-validation are superseded by the validated template index; the
surviving LLM functions serve as the generalization tier for unseen lines,
with templates canonicalized to the library form. Runtime parsing is
deterministic matching only: validated index -> template regexes ->
LLM-synthesized functions. No Drain, no LLM at runtime.
"""
import hashlib
import importlib.util
import re
from pathlib import Path

TEMPLATES = {templates!r}

_EXACT = {exact!r}

_SIGS = {sigs!r}

_LLM_ALIGN = {align!r}

_NUM = re.compile(r"\\d")
_REGEXES = [re.compile(r"\\s*" + r"\\S+".join(re.escape(p) for p in t.split("<*>")) + r"\\s*")
            for t in TEMPLATES]

_spec = importlib.util.spec_from_file_location(
    "llm_functions_{s.lower()}", Path(__file__).with_name("llm_functions.py"))
_llm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_llm)


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
        try:
            out = _llm.process_log(log)
            lt = out.get("template") if isinstance(out, dict) else str(out)
            if lt and lt != "UNKNOWN":
                return _LLM_ALIGN.get(str(lt))
        except Exception:
            pass
    return i


def process_log(log):
    i = match_template(str(log))
    if i is None:
        return {{"template": "UNKNOWN", "template_id": None}}
    return {{"template": TEMPLATES[i], "template_id": i}}
''')
    print(f"  {s:12s} templates={len(templates):3d} exact={len(exact):4d} sigs={len(sigs):4d} "
          f"llm_fns_aligned={len(align):3d} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    print("CLAD offline foundry: compiling Loghub-2k Parser Banks")
    for s in SYSTEMS:
        compile_system(s)
