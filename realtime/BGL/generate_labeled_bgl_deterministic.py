#!/usr/bin/env python3
"""
Generate Labeled BGL Dataset - DETERMINISTIC VERSION
=====================================================
Produces: labeled_logs_BGL.csv with columns [EventId, Label]

CRITICAL FIX: Uses sequential template matching (template 1 to 1848)
to ensure deterministic EventId assignment. The FIRST matching template
(lowest ID) is always selected.

This ensures consistency between training and inference.
"""

import os
import sys
import re
import time
import pandas as pd
from tqdm import tqdm

# Add training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

# Configuration
# Full raw BGL log; obtain with ../../data/download_full_datasets.sh
BGL_LOG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'full', 'BGL.log')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'labeled_logs_BGL_deterministic.csv')

# BGL log regex pattern
BGL_REGEX = re.compile(
    r'^(\S+)\s+'                          # Label
    r'(\d+)\s+'                          # Unix timestamp
    r'(\d+\.\d+\.\d+)\s+'               # Date
    r'(\S+)\s+'                          # Node
    r'(\d+-\d+-\d+-\d+\.\d+\.\d+\.\d+)\s+'  # Time
    r'(\S+)\s+'                          # NodeRepeat
    r'(\w+)\s+'                          # Type
    r'(\w+)\s+'                          # Component
    r'(\w+)\s*'                          # Level
    r'(.*)$'                             # Content
)


class DeterministicQwenLogParser:
    """DETERMINISTIC QwenLog parser - always selects LOWEST matching template ID"""
    
    def __init__(self):
        self.all_parsers = []
        self._load_parsers()
    
    def _load_parsers(self):
        """Load parser functions in ORDER (1 to 1848)"""
        try:
            import qwenlog_full_parser_bank as parser_bank
            
            print("Loading QwenLog BGL template functions...")
            loaded = 0
            
            for i in range(1, 2000):
                is_func_name = f'is_log_template_{i}'
                if hasattr(parser_bank, is_func_name):
                    is_func = getattr(parser_bank, is_func_name)
                    self.all_parsers.append((i, is_func))
                    loaded += 1
                elif loaded > 0 and i > loaded + 10:
                    break
            
            print(f"Loaded {loaded} template functions")
            
        except ImportError as e:
            print(f"Error loading QwenLog parser bank: {e}")
            raise
    
    def parse_content(self, content):
        """Parse content - ALWAYS returns FIRST (lowest ID) matching template"""
        for template_id, is_func in self.all_parsers:
            try:
                if is_func(content):
                    return f"Q{template_id}"
            except Exception:
                continue
        
        return "Q_UNKNOWN"


def parse_bgl_line(line, parser):
    """Parse a single BGL log line and return (EventId, Label)"""
    match = BGL_REGEX.match(line.strip())
    if not match:
        return None, None
    
    label_raw = match.group(1)
    content = match.group(10)
    
    # Determine label: "-" = normal (0), else anomaly (1)
    label = 0 if label_raw == "-" else 1
    
    # Get EventId from DETERMINISTIC parser
    event_id = parser.parse_content(content)
    
    return event_id, label


def main():
    """Generate labeled_logs_BGL_deterministic.csv"""
    print("\n" + "="*70)
    print(" GENERATE LABELED BGL DATASET (DETERMINISTIC)")
    print(" Always selects LOWEST matching template ID")
    print("="*70)
    
    # Check input file
    if not os.path.exists(BGL_LOG_PATH):
        print(f"ERROR: BGL log file not found at {BGL_LOG_PATH}")
        sys.exit(1)
    
    # Initialize DETERMINISTIC parser
    parser = DeterministicQwenLogParser()
    
    # Count total lines
    print("\nCounting lines...")
    with open(BGL_LOG_PATH, 'r', errors='ignore') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines:,}")
    
    # Process all lines
    print("\nParsing logs (DETERMINISTIC - lowest template ID wins)...")
    results = []
    
    start_time = time.time()
    
    with open(BGL_LOG_PATH, 'r', errors='ignore') as f:
        for line in tqdm(f, total=total_lines, desc="Processing"):
            event_id, label = parse_bgl_line(line, parser)
            if event_id is not None:
                results.append({'EventId': event_id, 'Label': label})
    
    elapsed = time.time() - start_time
    
    # Save to CSV
    print(f"\nSaving to {OUTPUT_PATH}...")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE (DETERMINISTIC)")
    print("="*70)
    print(f"Total logs: {len(results):,}")
    print(f"Time: {elapsed:.1f}s ({len(results)/elapsed:,.0f} logs/sec)")
    print(f"Anomaly ratio: {df['Label'].mean()*100:.2f}%")
    print(f"Unique EventIds: {df['EventId'].nunique()}")
    print(f"Output: {OUTPUT_PATH}")
    
    # Show EventId distribution
    print("\nTop 10 EventIds:")
    print(df['EventId'].value_counts().head(10))


if __name__ == "__main__":
    main()
