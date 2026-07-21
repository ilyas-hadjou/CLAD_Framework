# CLAD — Contextual Log Anomaly Detection

CLAD is a two-stage tool for log analysis in distributed systems. An LLM
(Qwen2.5-Coder-7B-Instruct) is used **offline** to synthesize one small Python
parsing function per log template, compiled into a *Parser Bank*; at runtime,
parsing is deterministic function matching with no LLM calls. A lightweight
Transformer classifier then labels sliding windows of parsed event IDs as
normal or anomalous in real time.

## Repository layout

```
clad-tool/
├── quickstart.py                  # end-to-end run on bundled data (offline, CPU-ok)
├── requirements.txt
├── data/
│   ├── loghub-2k/                 # bundled Loghub-2k logs (11 systems)
│   ├── samples/                   # small labeled BGL event stream for the quickstart
│   └── download_full_datasets.sh  # fetches full BGL / Thunderbird / HDFS from Zenodo
├── parser/
│   ├── banks/                     # pre-built Parser Banks (qwen2.5-7b/, qwen3-30b/)
│   ├── synthesis/
│   │   ├── generate_parser_banks.ipynb   # offline Parser Bank synthesis (GPU)
│   │   └── full_scale/                   # full-dataset trainers (BGL/HDFS/Thunderbird)
│   └── eval/
│       ├── run_parser_2k.py       # run a bank over the 11 systems
│       ├── score_parser.py        # score predictions against ground truth
│       └── scalability.py         # runtime-scaling experiment
├── classifier/
│   ├── train.ipynb                # train the Transformer classifier
│   ├── test.ipynb                 # evaluate a checkpoint (incl. noise-injection tests)
│   └── models/                    # pre-trained checkpoints
└── realtime/
    ├── BGL/                       # streaming pipeline: parse -> window -> classify
    └── Thunderbird/               # (train + real-time test notebooks, parser bank,
                                   #  checkpoint, vocabulary, per-folder README)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python ≥ 3.10, `torch 2.8.0` (pinned; the shipped checkpoints
were saved with it). GPU is optional for everything except Parser Bank
synthesis, which loads Qwen2.5-Coder-7B in FP16 (≈16 GB VRAM).

## Quick start

Runs entirely from bundled data — no downloads, CPU is sufficient:

```bash
python quickstart.py
```

Step 1 parses the 11 bundled Loghub-2k systems with the pre-built Parser Bank
and scores the predictions. Step 2 loads the pre-trained real-time BGL
classifier checkpoint and classifies the bundled labeled sample. The run
completes in a few minutes and prints its progress; it should finish with
`Quickstart finished.` and no errors.

## Running the pipeline on full datasets

```bash
# 1. download raw logs (BGL ≈ 700 MB, Thunderbird ≈ 30 GB unpacked)
bash data/download_full_datasets.sh bgl        # or: thunderbird | hdfs | all

# 2. produce a labeled event stream with the Parser Bank
python realtime/BGL/generate_labeled_bgl_deterministic.py

# 3. train and run the real-time detector
#    open and run: realtime/BGL/train_BGL_Classifer.ipynb
#                  realtime/BGL/test_BGL_realtime.ipynb
```

The Thunderbird pipeline is analogous under `realtime/Thunderbird/`
(`generate_labeled_thunderbird_chunks.py`, then the train/test notebooks).

To synthesize Parser Banks yourself, open
`parser/synthesis/generate_parser_banks.ipynb` and run all cells; regenerated
banks are written to `parser/synthesis/regenerated_output/`.

## Datasets

The bundled Loghub-2k samples are redistributed under the Loghub terms (see
`data/loghub-2k/LOGHUB_LICENSE.txt`). Full datasets are fetched from
[Zenodo record 8196385](https://zenodo.org/records/8196385); please cite the
Loghub paper when using them.
