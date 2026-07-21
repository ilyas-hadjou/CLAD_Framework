# Thunderbird Real-Time Anomaly Detection

LogInferNet real-time log anomaly detection for Thunderbird supercomputer logs. Achieves **99.54% F1 score** on 70.4M logs

##  Performance Highlights

| Metric | Value |
|--------|-------|
| **Dataset** | Thunderbird Chunk1 (70.4M logs) |
| **F1 Score** | **99.54%** |
| **Accuracy** | 99.79% |
| **Precision** | 99.86% |
| **Recall** | 99.23% |
| **Throughput** | 1,162 lines/sec |
| **Benchmark Time** | 16.83 hours (full 70M) |

##  Confusion Matrix (70.4M logs)

|  | Predicted Normal | Predicted Anomaly |
|---|---|---|
| **True Normal** | 54,174,802 (TN) | 22,878 (FP) |
| **True Anomaly** | 124,298 (FN) | **16,081,977 (TP)** |

- **False Positive Rate**: 0.04%
- **False Negative Rate**: 0.76%
- **True Positive Rate (Recall)**: 99.23%

## 📁 Project Structure

```
Thunderbird_Anomaly_Detection/
├── models/
│   ├── TableVIII_thunderbird_realtime_classifier.pth  # 2.5MB trained model, reproduces Table VIII (Thunderbird row)
│   ├── event_vocab_Thunderbird_Chunk1.json         # 609 unique EventIds
│   └── training_metadata_Thunderbird_Chunk1.json   # Model configuration
├── training/
│   └── qwenlog_thunderbird_parser_bank.py          # 6,663 parser templates
├── train_Thunderbird_Chunk1.ipynb                  # Training notebook
├── test_Thunderbird_Chunk1_realtime.ipynb          # Real-time benchmark
└── results/
    └── thunderbird_chunk1_full_benchmark.json      # Full benchmark results
```

##  Quick Start

### 1. Prerequisites

```bash
pip install torch pandas scikit-learn tqdm numpy
```

Requirements:
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### 2. Download Thunderbird Dataset

Download from [Loghub](https://github.com/logpai/loghub):

```bash
# Download (30GB uncompressed)
mkdir -p qwenlog_final/dataset
cd qwenlog_final/dataset
wget https://zenodo.org/record/3227177/files/Thunderbird.tar.gz
tar -xzf Thunderbird.tar.gz
cd ../..
```

Dataset info:
- **Source**: LANL Thunderbird supercomputer
- **Total logs**: 211,212,192
- **Anomaly rate**: 1.54%
- **Format**: Timestamp + Node + Alert Type + Message

### 3. Run Real-Time Detection

```bash
jupyter notebook test_Thunderbird_Chunk1_realtime.ipynb
```

The notebook will:
1. Load the trained model
2. Initialize the QwenLog parser (6,663 templates)
3. Process 70.4M logs in real-time
4. Report F1, accuracy, and throughput

**Expected runtime**: ~17 hours on CPU, ~8 hours on GPU

### 4. Training (Optional)

To train from scratch:

```bash
jupyter notebook train_Thunderbird_Chunk1.ipynb
```

Training requirements:
- 70.4M labeled logs (generated from raw logs)
- ~16 hours training time (5 epochs)
- ~8GB GPU memory

## 📈 Model Architecture

### Transformer-based Sequence Classifier

```python
class EventSequenceModel(nn.Module):
    - Embedding Layer: vocab_size=609, embed_dim=64
    - Position Encoding: Learnable embeddings
    - Transformer Encoder: 2 layers, 4 attention heads, FFN dim=128
    - Attention Pooling: Weighted sequence aggregation
    - Classifier: 2-layer MLP (64 → 128 → 2)
    
Total Parameters: 648,899
Model Size: 2.5MB
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 609 EventIds |
| Embedding Dimension | 64 |
| Hidden Dimension | 128 |
| Attention Heads | 4 |
| Transformer Layers | 2 |
| Sequence Length | 80 events |
| Block Size | 500 lines |
| Dropout | 0.3 |
| Batch Size | 64 |
| Epochs | 5 |

##  Training Details

### Dataset Preparation

**Chunk 1 Statistics**:
- Total logs: 70,404,049
- Anomaly logs: 1,821,472 (2.59%)
- Unique EventIds: 609
- Training sequences: 47,321,472

**Data Generation**:
```bash
python generate_labeled_thunderbird_chunks.py --chunk 1
```

This creates `thunderbird_chunks/labeled_logs_thunderbird_chunk1.csv` with:
- EventId (deterministic template matching)
- Label (0=normal, 1=anomaly)
- LineId (original line number)

### Training Process

**Configuration**:
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Training sequences: 47.3M
- Batches per epoch: 739,240
- Training time: ~16 hours (5 epochs)

**Training Accuracy per Epoch**:
| Epoch | Accuracy | Loss |
|-------|----------|------|
| 1 | 99.12% | 0.0287 |
| 2 | 99.56% | 0.0142 |
| 3 | 99.71% | 0.0095 |
| 4 | 99.77% | 0.0073 |
| 5 | **99.79%** | 0.0065 |

## 📊 Benchmark Results

### Full Chunk1 Real-Time Benchmark (70.4M logs)

**Dataset**:
- Total lines: 70,404,049
- Total sequences: 70,403,955
- Raw anomaly lines: 1,821,472 (2.59%)

**Detection Results**:
- True anomaly sequences: 16,206,275
- Predicted anomaly sequences: 16,104,855

**Performance**:
- Total time: 16.83 hours (60,603 seconds)
- Throughput: 1,162 lines/sec
- Sequences/sec: 1,162

**Metrics**:
- Accuracy: 99.79%
- F1 Score: **99.54%**
- Precision: 99.86%
- Recall: 99.23%

### Comparison: Real-Time vs Pre-Parsed

| Mode | Throughput | F1 Score | Note |
|------|------------|----------|------|
| Real-Time | 1,162 lines/s | 99.54% | Parse + Classify |
| Pre-Parsed | ~19K seq/s | 99.79% | Classify only |

Real-time mode includes:
1. Regex parsing of Thunderbird log format
2. Sequential template matching (6,663 templates)
3. Vocabulary mapping
4. Transformer inference

## 🧪 Parser Details

### QwenLog Template Bank

**Characteristics**:
- Total templates: 6,663
- Generated by: Qwen2.5-7B-Instruct
- Matching strategy: Sequential (template 1 → 6663, first match wins)
- Deterministic: Same input always produces same EventId

**Template Example**:
```python
def is_log_template_1(content):
    # Template: "instruction cache parity error corrected"
    pattern = r'^instruction\s+cache\s+parity\s+error\s+corrected$'
    return bool(re.match(pattern, content, re.IGNORECASE))
```

**Parsing Pipeline**:
1. Extract content from Thunderbird format (regex)
2. Try templates sequentially (1→6663)
3. First match wins → return EventId
4. No match → return UNKNOWN token

**Coverage**:
- Known events: 609 unique EventIds (from 6,663 templates)
- Unknown rate: <0.1% in Chunk1

## 📋 Results Files

### `thunderbird_chunk1_full_benchmark.json`

```json
{
  "dataset": "Thunderbird_Chunk1",
  "benchmark_type": "full_real_time",
  "total_lines": 70404049,
  "metrics": {
    "accuracy": 0.9979,
    "f1_score": 0.9954,
    "precision": 0.9986,
    "recall": 0.9923
  },
  "confusion_matrix": {
    "TN": 54174802,
    "FP": 22878,
    "FN": 124298,
    "TP": 16081977
  },
  "performance": {
    "total_time_hours": 16.83,
    "throughput_lines_per_sec": 1162
  }
}
```

##  Error Analysis

### False Positives (22,878 instances, 0.04%)

Likely causes:
- Rare event patterns flagged as anomalies
- EventIds near decision boundary
- Template matching edge cases

### False Negatives (124,298 instances, 0.76%)

Likely causes:
- Anomaly patterns similar to normal sequences
- Insufficient context in 80-event window
- Class imbalance (97.4% normal vs 2.6% anomaly)

##  Technical Approach

### Why Thunderbird is Challenging

1. **Scale**: 211M logs (30GB raw data)
2. **Sparsity**: Only 1.54% anomalies overall
3. **Complexity**: 6,663 unique log templates
4. **Chunk Strategy**: Split into 3×70M for manageable training

### Key Innovations

1. **Deterministic Parsing**: No LLM calls at inference time
2. **Sequential Template Matching**: O(n) worst case, early exit on match
3. **Sliding Window**: 80-event sequences with 500-line blocks
4. **Balanced Sampling**: Handles 97:3 class imbalance

### Reproducibility

All components are deterministic:
- ✅ Parser: Sequential template matching (no randomness)
- ✅ Vocabulary: Fixed mapping from EventId to index
- ✅ Model: Set random seeds for training
- ✅ Evaluation: Fixed test set (first 70M lines)

## 🚀 Future Work

### Chunk 2 & 3 Training

- **Chunk 2**: Lines 70M-140M (502 unique EventIds)
- **Chunk 3**: Lines 140M-211M (426 unique EventIds)

### Model Improvements

- [ ] Increase sequence length (80 → 128 events)
- [ ] Add LSTM/GRU layer for longer context
- [ ] Experiment with RoBERTa-style pretraining
- [ ] Multi-task learning (template + anomaly)

### Deployment

- [ ] ONNX export for faster inference
- [ ] Quantization (FP32 → INT8) for edge devices
- [ ] Real-time streaming pipeline (Kafka/Flink)
- [ ] Online learning for concept drift

##  Dataset Citation

```bibtex
@inproceedings{oliner2007supercomputers,
  title={Supercomputers as Blackboxes: Machine Learning for System Diagnosis},
  author={Oliner, Adam and Stearley, Jon},
  booktitle={USENIX Workshop on Hot Topics in System Dependability},
  year={2007}
}
```

Dataset source: [Loghub - Thunderbird](https://github.com/logpai/loghub)

## 📄 License

ISSLAB KNU 

## 🙏 Acknowledgments

- **Dataset**: LANL Thunderbird logs via Loghub
- **Parser Generation**: Qwen2.5-7B-Instruct
- **Classifer**: Seqnet Dr.Irshad)
- **Framework**: PyTorch, scikit-learn
- **Computing**: Training performed on NVIDIA GPU cluster

---

**Last Updated**: December 2025  
**Model Version**: v1.0-chunk1  
**Status**: ✅ Production Ready
