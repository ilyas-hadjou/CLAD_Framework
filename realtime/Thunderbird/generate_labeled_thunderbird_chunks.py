#!/usr/bin/env python3
"""
Generate deterministic labeled data for Thunderbird dataset
Split into 3 chunks for separate training/testing
"""

import re
import sys
import os
from tqdm import tqdm

# Add training directory to path
sys.path.insert(0, './training')
import qwenlog_thunderbird_parser_bank as parser_bank

# Configuration
# Full raw Thunderbird log; obtain with ../../data/download_full_datasets.sh
INPUT_FILE = "../../data/full/Thunderbird.log"
OUTPUT_DIR = "./thunderbird_chunks"
TOTAL_LINES = 211212192
CHUNK_SIZE = TOTAL_LINES // 3  # ~70M logs per chunk

# Thunderbird log regex
# Format: Label Timestamp Date Node Time User Content
# Example: - 1131523501 2005.11.09 aadmin1 Nov 10 00:05:01 src@aadmin1 message...
THUNDERBIRD_REGEX = re.compile(
    r'^(\S+)\s+'           # Label (- or alert type like ECC, VAPI, etc.)
    r'(\d+)\s+'            # Timestamp
    r'(\d+\.\d+\.\d+)\s+'  # Date
    r'(\S+)\s+'            # Node
    r'(\w+\s+\d+)\s+'      # Month Day
    r'(\d+:\d+:\d+)\s+'    # Time
    r'(\S+)\s+'            # User
    r'(.*)$'               # Content
)

def load_parsers():
    """Load all parser functions sequentially (deterministic order)"""
    parsers = []
    for i in range(1, 10000):  # Thunderbird has 6663 templates
        if hasattr(parser_bank, f'is_log_template_{i}'):
            parsers.append((i, getattr(parser_bank, f'is_log_template_{i}')))
    print(f"Loaded {len(parsers)} parser templates")
    return parsers

def parse_line(content, parsers):
    """Parse content using deterministic sequential matching"""
    for tid, is_func in parsers:
        try:
            if is_func(content):
                return f"Q{tid}"
        except:
            pass
    return "Q_UNKNOWN"

def process_chunk(chunk_id, start_line, end_line, parsers):
    """Process a chunk of the dataset"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"labeled_logs_thunderbird_chunk{chunk_id}.csv")
    
    print(f"\n{'='*60}")
    print(f"Processing Chunk {chunk_id}: lines {start_line:,} to {end_line:,}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    event_vocab = set()
    stats = {'total': 0, 'parsed': 0, 'normal': 0, 'anomaly': 0}
    
    with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        # Write header
        fout.write("Content,EventId,Label\n")
        
        # Skip to start line
        for _ in tqdm(range(start_line), desc=f"Skipping to line {start_line:,}", unit="line"):
            fin.readline()
        
        # Process chunk
        for _ in tqdm(range(end_line - start_line), desc=f"Chunk {chunk_id}", unit="line"):
            line = fin.readline()
            if not line:
                break
            
            stats['total'] += 1
            match = THUNDERBIRD_REGEX.match(line.strip())
            
            if not match:
                continue
            
            label_str = match.group(1)
            content = match.group(8)
            
            # Label: 0 = normal (-), 1 = anomaly (anything else)
            label = 0 if label_str == '-' else 1
            
            # Parse content to get EventId
            event_id = parse_line(content, parsers)
            event_vocab.add(event_id)
            
            # Escape content for CSV
            content_escaped = content.replace('"', '""')
            fout.write(f'"{content_escaped}",{event_id},{label}\n')
            
            stats['parsed'] += 1
            if label == 0:
                stats['normal'] += 1
            else:
                stats['anomaly'] += 1
    
    # Save vocab for this chunk
    vocab_file = os.path.join(OUTPUT_DIR, f"event_vocab_thunderbird_chunk{chunk_id}.json")
    import json
    vocab_dict = {eid: idx for idx, eid in enumerate(sorted(event_vocab))}
    with open(vocab_file, 'w') as f:
        json.dump(vocab_dict, f, indent=2)
    
    print(f"\nChunk {chunk_id} Statistics:")
    print(f"  Total lines: {stats['total']:,}")
    print(f"  Parsed: {stats['parsed']:,}")
    print(f"  Normal: {stats['normal']:,} ({100*stats['normal']/stats['parsed']:.2f}%)")
    print(f"  Anomaly: {stats['anomaly']:,} ({100*stats['anomaly']/stats['parsed']:.2f}%)")
    print(f"  Unique EventIds: {len(event_vocab)}")
    print(f"  Vocab saved: {vocab_file}")
    
    return event_vocab, stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Thunderbird labeled data in chunks')
    parser.add_argument('--chunk', type=int, choices=[1, 2, 3, 0], default=0,
                        help='Chunk to process (1, 2, 3) or 0 for all')
    args = parser.parse_args()
    
    print("="*60)
    print("Thunderbird Deterministic Labeling - Chunked Processing")
    print("="*60)
    print(f"Input: {INPUT_FILE}")
    print(f"Total lines: {TOTAL_LINES:,}")
    print(f"Chunk size: ~{CHUNK_SIZE:,} lines each")
    print()
    
    # Load parsers once
    parsers = load_parsers()
    
    # Define chunks
    chunks = {
        1: (0, CHUNK_SIZE),
        2: (CHUNK_SIZE, 2 * CHUNK_SIZE),
        3: (2 * CHUNK_SIZE, TOTAL_LINES)
    }
    
    all_vocab = set()
    all_stats = {'total': 0, 'parsed': 0, 'normal': 0, 'anomaly': 0}
    
    chunks_to_process = [args.chunk] if args.chunk > 0 else [1, 2, 3]
    
    for chunk_id in chunks_to_process:
        start, end = chunks[chunk_id]
        vocab, stats = process_chunk(chunk_id, start, end, parsers)
        all_vocab.update(vocab)
        for k in all_stats:
            all_stats[k] += stats[k]
    
    if len(chunks_to_process) > 1:
        print("\n" + "="*60)
        print("Overall Statistics (All Chunks)")
        print("="*60)
        print(f"  Total lines: {all_stats['total']:,}")
        print(f"  Parsed: {all_stats['parsed']:,}")
        print(f"  Normal: {all_stats['normal']:,}")
        print(f"  Anomaly: {all_stats['anomaly']:,}")
        print(f"  Combined unique EventIds: {len(all_vocab)}")
        
        # Save combined vocab
        import json
        vocab_dict = {eid: idx for idx, eid in enumerate(sorted(all_vocab))}
        combined_vocab_file = os.path.join(OUTPUT_DIR, "event_vocab_thunderbird_combined.json")
        with open(combined_vocab_file, 'w') as f:
            json.dump(vocab_dict, f, indent=2)
        print(f"  Combined vocab saved: {combined_vocab_file}")

if __name__ == "__main__":
    main()
