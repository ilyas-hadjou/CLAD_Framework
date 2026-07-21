# Real-Time Log Anomaly Detection

LLM-generated log parser with real-time anomaly detection.

##  Features

- **GenLog Parser**: LLM-generated templates for deterministic log parsing
- **Real-Time Pipeline**: Parse + classify logs on-the-fly (~25K seq/s)
- **Transformer Classifier**: Sequence-based anomaly detection (98.35% F1)

##  Results

| Method | Throughput | Accuracy | F1 Score |
|--------|------------|----------|----------|
| Real-Time (GenLog) | 24,911 seq/s | 99.73% | 98.35% |
| Pre-Parsed (DataLoader) | 28,665 seq/s | 99.86% | 99.29% |

## 📁 Project Structure

```
├── test_BGL_realtime.ipynb      # Benchmark notebook (Real-Time vs Pre-Parsed)
├── train_BGL_Classifer.ipynb    # Training notebook
├── generate_labeled_bgl_deterministic.py  # Data generation script
├── training/
│   └── qwenlog_full_parser_bank.py  # 1,848 LLM-generated parser templates
├── models/
│   ├── TableVIII_bgl_realtime_classifier.pth  # Trained model (2.3MB), reproduces Table VIII (BGL row)
│   ├── training_metadata_BGL.json   # Model config
│   └── event_vocab_BGL.json         # Event vocabulary (99 templates)
└── results/
    └── qwenlog_comparison_*.json    # Benchmark results
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch pandas scikit-learn tqdm
```

### 2. Download BGL Dataset
Download from [Loghub](https://github.com/logpai/loghub) (~709MB):
```bash
mkdir -p dataset
wget https://zenodo.org/record/3227177/files/BGL.tar.gz
tar -xzf BGL.tar.gz -C dataset/
```

### 3. Generate Labeled Data
```bash
python generate_labeled_bgl_deterministic.py
```
This creates `labeled_logs_BGL_deterministic.csv` (~24MB, ~5min to generate).

## 🏃 Running the Benchmark

Open and run `test_BGL_realtime.ipynb`:

1. **Cells 1-5**: Setup (imports, model, parser, pipeline)
2. **Cells 6-8**: Real-time benchmark (parses raw logs on-the-fly)
3. **Cells 9-12**: Pre-parsed comparison (uses pre-generated CSV)

### Expected Output
```
COMPARISON: REAL-TIME vs PRE-PARSED
======================================================================
Metric                    Real-Time            Pre-Parsed          
----------------------------------------------------------------------
Throughput (seq/s)        24,911               28,665              
Accuracy (%)              99.73                99.86               
F1 Score (%)              98.35                99.29               
```

##  Notes

- **GPU recommended**: RTX 4090 achieves ~25K seq/s throughput
- **Real-time mode**: Parses each log line → extracts EventId → classifies
- **Pre-parsed mode**: Uses pre-generated CSV (faster, no parsing overhead)
- **Templates**: 1,848 in parser bank, 99 unique EventIds observed in BGL

##  License
ISSLAB KNU 
