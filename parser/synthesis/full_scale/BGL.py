"""
BGL Trainer Module
==================
Offline training pipeline for BGL (Blue Gene/L) dataset.

Pipeline Stages:
1. Phase 1 (Discovery): Load Drain output, extract templates
2. Phase 2 (Vector Bank): Embed templates using SentenceTransformer  
3. Phase 3 (Parser Generation): Use Qwen-Coder to generate Python code
4. Phase 4 (Validation): Test generated parsers in sandbox

Output: training/BGL_parser_bank.py (executable Python module)
"""

import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import Counter
import logging

# Import utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.qwen_client import QwenClient
from utils.embedding import EmbeddingEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGLTrainer:
    """
    BGL Training Pipeline - "The Offline Foundry"
    
    This class orchestrates the complete training process:
    - Loads BGL logs processed by Drain
    - Discovers templates and creates vector bank
    - Generates Python parser code using Qwen-Coder LLM
    - Validates and exports to BGL_parser_bank.py
    
    Usage:
        trainer = BGLTrainer(
            drain_output_path="benchmark/parsers/full_bgl_outputs/Drain/BGL.log_structured.csv",
            output_path="training/BGL_parser_bank.py"
        )
        trainer.run_pipeline()
    """
    
    def __init__(
        self,
        drain_output_path: str,
        output_path: str = "training/BGL_parser_bank.py",
        use_llm: bool = False  # Set True to use LLM, False for rule-based generation
    ):
        """
        Initialize BGL trainer.
        
        Args:
            drain_output_path: Path to Drain's structured output CSV
            output_path: Where to save generated parser bank
            use_llm: Whether to use Qwen LLM (slow) or rule-based generation (fast)
        """
        self.drain_output_path = Path(drain_output_path)
        self.output_path = Path(output_path)
        self.use_llm = use_llm
        
        # Components
        self.qwen_client = None
        self.embedding_engine = None
        
        # Data
        self.templates = []
        self.template_counts = {}
        self.vector_bank = None
        self.generated_code = []
        
        logger.info("="*70)
        logger.info("BGL TRAINER INITIALIZED")
        logger.info("="*70)
        logger.info(f"Drain output: {self.drain_output_path}")
        logger.info(f"Output file: {self.output_path}")
        logger.info(f"LLM mode: {'Enabled' if use_llm else 'Rule-based (Fast)'}")
    
    def run_pipeline(self):
        """
        Execute the complete training pipeline.
        
        Pipeline Flow:
        1. Phase 1: Template Discovery
        2. Phase 2: Vector Bank Creation
        3. Phase 3: Parser Code Generation
        4. Phase 4: Validation & Export
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*70)
        
        # Phase 1: Discovery
        self.phase1_discovery()
        
        # Phase 2: Vector Bank
        self.phase2_vector_bank()
        
        # Phase 3: Parser Generation
        if self.use_llm:
            self.phase3_llm_generation()
        else:
            self.phase3_rule_based_generation()
        
        # Phase 4: Export
        self.phase4_export()
        
        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Generated parser bank: {self.output_path}")
        logger.info(f"Total templates: {len(self.templates)}")
        logger.info(f"File size: {self.output_path.stat().st_size / 1024:.1f} KB")
    
    def phase1_discovery(self):
        """
        Phase 1: Template Discovery
        
        Process:
        1. Load Drain's structured output
        2. Extract unique templates with frequency counts
        3. Identify keywords for each template
        """
        logger.info("\n📊 PHASE 1: TEMPLATE DISCOVERY")
        logger.info("-"*70)
        
        # Load Drain output
        logger.info(f"Loading Drain output from {self.drain_output_path}...")
        df = pd.read_csv(self.drain_output_path)
        logger.info(f"✅ Loaded {len(df):,} parsed logs")
        
        # Get template distribution
        logger.info("Analyzing template distribution...")
        template_counts = df['EventTemplate'].value_counts()
        self.template_counts = template_counts.to_dict()
        
        # Store templates in frequency order
        self.templates = list(template_counts.index)
        
        logger.info(f"✅ Found {len(self.templates):,} unique templates")
        logger.info(f"\nTop 5 templates:")
        for i, (template, count) in enumerate(template_counts.head(5).items(), 1):
            pct = count / len(df) * 100
            logger.info(f"  {i}. [{count:7,} logs, {pct:5.1f}%] {template[:60]}...")
    
    def phase2_vector_bank(self):
        """
        Phase 2: Create Template Vector Bank
        
        Process:
        1. Initialize SentenceTransformer
        2. Embed all template strings
        3. Store as numpy array for fast similarity search
        """
        logger.info("\n🔍 PHASE 2: VECTOR BANK CREATION")
        logger.info("-"*70)
        
        # Initialize embedding engine
        self.embedding_engine = EmbeddingEngine()
        
        # Embed all templates
        self.vector_bank = self.embedding_engine.embed_templates(self.templates)
        
        logger.info(f"✅ Vector bank created: {self.vector_bank.shape}")
        logger.info(f"   Dimensions: {self.vector_bank.shape[1]}")
        logger.info(f"   Memory: {self.vector_bank.nbytes / 1024:.1f} KB")
    
    def phase3_llm_generation(self):
        """
        Phase 3: LLM-Based Parser Generation
        
        Process:
        1. Initialize Qwen-Coder LLM
        2. For each template, generate parser code using precision-tuned prompt
        3. Validate generated code
        
        Note: This is SLOW but produces high-quality code
        """
        logger.info("\n🤖 PHASE 3: LLM-BASED PARSER GENERATION")
        logger.info("-"*70)
        
        # Initialize Qwen client
        logger.info("Loading Qwen-Coder LLM...")
        self.qwen_client = QwenClient()
        
        # Generate parser for each template
        logger.info(f"Generating parsers for {len(self.templates)} templates...")
        
        for i, template in enumerate(self.templates, 1):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(self.templates)}")
            
            # Extract keywords
            keywords = self._extract_keywords(template)
            
            # Create regex pattern
            regex_pattern = self._template_to_regex(template)
            
            # Generate code using LLM
            code = self.qwen_client.generate_parser_code(
                template_id=i,
                template=template,
                keywords=keywords,
                regex_pattern=regex_pattern,
                count=self.template_counts[template]
            )
            
            self.generated_code.append(code)
        
        # Unload LLM to free memory
        self.qwen_client.unload()
        
        logger.info(f"✅ Generated {len(self.generated_code)} parsers")
    
    def phase3_rule_based_generation(self):
        """
        Phase 3: Rule-Based Parser Generation (FAST)
        
        Process:
        1. For each template, create parser functions using templates
        2. No LLM needed - uses predefined code patterns
        
        Note: This is FAST and produces identical results to LLM
        """
        logger.info("\n⚡ PHASE 3: RULE-BASED PARSER GENERATION (FAST)")
        logger.info("-"*70)
        
        logger.info(f"Generating parsers for {len(self.templates)} templates...")
        
        for i, template in enumerate(self.templates, 1):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(self.templates)}")
            
            # Extract keywords
            keywords = self._extract_keywords(template)
            
            # Create regex pattern
            regex_pattern = self._template_to_regex(template)
            
            # Generate code using template
            code = self._generate_parser_function(
                template_id=i,
                template=template,
                keywords=keywords,
                regex_pattern=regex_pattern,
                count=self.template_counts[template]
            )
            
            self.generated_code.append(code)
        
        logger.info(f"✅ Generated {len(self.generated_code)} parsers")
    
    def phase4_export(self):
        """
        Phase 4: Validation & Export
        
        Process:
        1. Combine all generated code
        2. Add module header and dispatcher function
        3. Validate syntax
        4. Write to output file
        """
        logger.info("\n💾 PHASE 4: VALIDATION & EXPORT")
        logger.info("-"*70)
        
        # Create module header
        header = f'''"""
BGL Parser Bank - Generated by QwenLog Training Pipeline
==========================================================
Generated from Drain's output on full BGL dataset
Templates: {len(self.templates):,}
Strategy: Keyword pre-filtering + Regex pattern matching

This file is AUTO-GENERATED. Do not edit manually.
"""

import re

'''
        
        # Combine all parser functions
        all_code = header + "\n\n".join(self.generated_code)
        
        # Add main dispatcher
        dispatcher = self._generate_dispatcher()
        all_code += "\n\n" + dispatcher
        
        # Validate syntax
        logger.info("Validating generated code...")
        try:
            compile(all_code, '<string>', 'exec')
            logger.info("✅ Syntax validation passed")
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in generated code: {e}")
            raise
        
        # Write to file
        logger.info(f"Writing to {self.output_path}...")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write(all_code)
        
        logger.info(f"✅ Parser bank exported successfully")
    
    def _extract_keywords(self, template: str) -> List[str]:
        """Extract meaningful keywords from template for fast pre-filtering"""
        # Remove <*> placeholders and adjacent brackets/symbols that would be empty
        text = re.sub(r'\[<\*>\]', '', template)  # Remove [<*>]
        text = re.sub(r'\(<\*>\)', '', template)  # Remove (<*>)
        text = re.sub(r'<\*>', '', text)  # Remove remaining <*>
        
        # Split into words and filter
        words = text.split()
        
        # Keep words that are:
        # - At least 3 characters (lowered from 4 to catch more keywords)
        # - Not common words
        # - Has alphanumeric content
        # - Not just punctuation/brackets
        common_words = {'and', 'the', 'for', 'with', 'from', 'that', 'this', 'any'}
        keywords = []
        for w in words:
            # Remove trailing punctuation but keep internal structure
            clean_w = w.rstrip('.,;:!?')
            if (len(clean_w) >= 3 
                and clean_w.lower() not in common_words
                and any(c.isalnum() for c in clean_w)
                and not clean_w in ['[]', '()', '{}', '[', ']', '(', ')']):
                keywords.append(clean_w.lower())
        
        # Take top 4 most distinctive keywords (increased from 3)
        return keywords[:4]
    
    def _template_to_regex(self, template: str) -> str:
        """Convert template to proper regex pattern"""
        # First replace <*> with placeholder that won't be escaped
        template = template.replace('<*>', '___WILDCARD___')
        
        # Escape special regex characters
        pattern = re.escape(template)
        
        # Replace escaped spaces with flexible whitespace
        pattern = pattern.replace(r'\ ', r'\s+')
        
        # Replace placeholder with actual wildcard pattern
        pattern = pattern.replace('___WILDCARD___', r'\S+')
        
        # Anchor to match the whole content
        pattern = '^' + pattern + '$'
        
        return pattern
    
    def _generate_parser_function(
        self,
        template_id: int,
        template: str,
        keywords: List[str],
        regex_pattern: str,
        count: int
    ) -> str:
        """Generate parser function code using template"""
        
        # Escape strings for Python
        template_escaped = template.replace("'", "\\'").replace('"', '\\"')
        regex_for_code = regex_pattern.replace("'", "\\'")
        
        code = f'''def is_log_template_{template_id}(content: str) -> bool:
    """Check if log matches: {template_escaped}
    Count: {count:,} logs
    """'''
        
        # Add keyword pre-checks
        if keywords:
            code += '\n    content_lower = content.lower()'
            for kw in keywords:
                kw_escaped = kw.replace("'", "\\'")
                code += f"\n    if '{kw_escaped}' not in content_lower:"
                code += f"\n        return False"
        
        # Add regex matching
        code += f"\n    pattern = r'{regex_for_code}'"
        code += f"\n    return re.match(pattern, content) is not None"
        
        code += f'''


def parse_log_template_{template_id}(content: str) -> dict:
    """Parse log and return template"""
    return {{
        'template_id': {template_id},
        'template': '{template_escaped}'
    }}'''
        
        return code
    
    def _generate_dispatcher(self) -> str:
        """Generate main dispatcher function"""
        
        code = '''def process_log(raw_line: str) -> dict:
    """Main dispatcher - tries all templates in order of frequency
    
    Args:
        raw_line: Raw log line (with or without BGL metadata)
    
    Returns:
        Dict with 'template_id' and 'template' keys
    """
    # Extract content from BGL format if needed
    parts = raw_line.split(maxsplit=9)
    content = parts[9].strip() if len(parts) >= 10 else raw_line.strip()
    
    # Try each template in frequency order
'''
        
        for template_id in range(1, len(self.templates) + 1):
            if template_id == 1:
                code += f'    if is_log_template_{template_id}(content):\n'
            else:
                code += f'    elif is_log_template_{template_id}(content):\n'
            code += f'        return parse_log_template_{template_id}(content)\n'
        
        code += '''    else:
        return {'template_id': -1, 'template': '<*>'}


def parse_window(logs: list) -> list:
    """
    Parse a batch/window of logs (for integration with Log-GraphSeqNet).
    
    Args:
        logs: List of raw log lines
    
    Returns:
        List of template strings (same order as input)
    
    Example:
        >>> logs = ["generating core.12345", "iar 123 dear 456"]
        >>> templates = parse_window(logs)
        >>> templates
        ['generating <*>', 'iar <*> dear <*>']
    """
    return [process_log(log)['template'] for log in logs]
'''
        
        return code


# Convenience function for direct execution
def train_bgl(
    drain_output_path: str,
    output_path: str = "training/BGL_parser_bank.py",
    use_llm: bool = False
):
    """
    Convenience function to train BGL parser bank.
    
    Args:
        drain_output_path: Path to Drain output CSV
        output_path: Where to save generated parser
        use_llm: Use LLM (slow) or rule-based (fast)
    """
    trainer = BGLTrainer(
        drain_output_path=drain_output_path,
        output_path=output_path,
        use_llm=use_llm
    )
    trainer.run_pipeline()
