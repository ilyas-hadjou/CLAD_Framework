"""
Qwen Client Module
==================
Wrapper for Qwen2.5-Coder-7B-Instruct LLM (Offline Code Generation)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class QwenClient:
    """
    Wrapper for Qwen-Coder LLM.
    
    Responsibilities:
    - Load Qwen2.5-Coder-7B-Instruct model
    - Generate Python parser code from precision-tuned prompts
    - Handle batched generation for efficiency
    
    Usage:
        client = QwenClient(model_path="Qwen/Qwen2.5-Coder-7B-Instruct")
        code = client.generate_parser(template, keywords, regex_pattern)
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device: str = "cuda",
        load_in_fp16: bool = True
    ):
        """
        Initialize Qwen client.
        
        Args:
            model_path: HuggingFace model identifier or local path
            device: 'cuda' or 'cpu'
            load_in_fp16: Use FP16 to save memory (14GB vs 28GB)
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing QwenClient: {model_path}")
        self._load_model(load_in_fp16)
    
    def _load_model(self, fp16: bool = True):
        """Load Qwen model and tokenizer"""
        logger.info(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        logger.info(f"Loading model (FP16={fp16})...")
        dtype = torch.float16 if fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        logger.info("✅ Qwen model loaded successfully")
    
    def generate_parser_code(
        self,
        template_id: int,
        template: str,
        keywords: List[str],
        regex_pattern: str,
        count: int
    ) -> str:
        """
        Generate Python parser function using precision-tuned prompt.
        
        This is the CRITICAL v5 feature: The prompt constrains the LLM
        to only generate code that extracts variables at <*> positions.
        
        Args:
            template_id: Unique template identifier
            template: Template string (e.g., "generating <*>")
            keywords: Keywords for pre-filtering
            regex_pattern: Compiled regex pattern
            count: Frequency count (for documentation)
        
        Returns:
            Python code string with is_log_template_N() and parse_log_template_N()
        """
        
        # Escape strings for Python code generation
        template_escaped = template.replace("'", "\\'").replace('"', '\\"')
        regex_escaped = regex_pattern.replace("'", "\\'")
        keywords_str = ", ".join([f"'{kw}'" for kw in keywords])
        
        # PRECISION-TUNED PROMPT v5.2 - ULTRA HIGH PA (95%+)
        prompt = f"""You are an expert Python code generator for log parsing. Generate EXACT matching parser functions.

**TARGET TEMPLATE (ID {template_id}):**
Template: {template}
Keywords: {keywords}
Regex Pattern: {regex_pattern}
Frequency: {count:,} logs

**ULTRA-PRECISION REQUIREMENTS (PA 95%+):**
1. Match EXACT template format with proper spacing
2. <*> wildcard = .+? (non-greedy) or \\S+ for single words
3. Escape ALL regex special chars: ( ) [ ] {{ }} . * + ? ^ $ | \\
4. Keywords are case-insensitive, but template structure is EXACT
5. Use anchored regex (^ and $) for full line matching
6. Return EXACT template string byte-for-byte

**FEW-SHOT EXAMPLES (95%+ PA):**

Example 1 - Simple template with special chars:
Template: "session opened for user <*> by (uid=<*>)"

```python
def is_log_template_1(content: str) -> bool:
    \"\"\"Check: session opened for user <*> by (uid=<*>)\"\"\"
    content_lower = content.lower()
    if 'session' not in content_lower:
        return False
    if 'opened' not in content_lower:
        return False
    if 'user' not in content_lower:
        return False
    if 'uid' not in content_lower:
        return False
    # EXACT regex with escaped parentheses
    pattern = r'^session opened for user .+? by \\(uid=.+?\\)$'
    return re.match(pattern, content) is not None

def parse_log_template_1(content: str) -> dict:
    return {{'template_id': 1, 'template': 'session opened for user <*> by (uid=<*>)'}}
```

Example 2 - Multiple wildcards with numbers:
Template: "PacketResponder <*> for block blk_<*> terminating"

```python
def is_log_template_2(content: str) -> bool:
    \"\"\"Check: PacketResponder <*> for block blk_<*> terminating\"\"\"
    content_lower = content.lower()
    if 'packetresponder' not in content_lower:
        return False
    if 'block' not in content_lower:
        return False
    if 'terminating' not in content_lower:
        return False
    # EXACT spacing and structure
    pattern = r'^PacketResponder .+? for block blk_.+? terminating$'
    return re.match(pattern, content) is not None

def parse_log_template_2(content: str) -> dict:
    return {{'template_id': 2, 'template': 'PacketResponder <*> for block blk_<*> terminating'}}
```

Example 3 - Brackets and dots:
Template: "Checking status of <*> [<*>.<*>]"

```python
def is_log_template_3(content: str) -> bool:
    \"\"\"Check: Checking status of <*> [<*>.<*>]\"\"\"
    content_lower = content.lower()
    if 'checking' not in content_lower:
        return False
    if 'status' not in content_lower:
        return False
    # Escape brackets and dots
    pattern = r'^Checking status of .+? \\[.+?\\..+?\\]$'
    return re.match(pattern, content) is not None

def parse_log_template_3(content: str) -> dict:
    return {{'template_id': 3, 'template': 'Checking status of <*> [<*>.<*>]'}}
```

**NOW GENERATE FOR TEMPLATE ID {template_id}:**
Generate code that matches the EXACT template structure with proper regex escaping.

```python
def is_log_template_{template_id}(content: str) -> bool:
    \"\"\"Check if log matches: {template_escaped}
    Count: {count:,} logs
    \"\"\"
    content_lower = content.lower()
    # Keyword pre-filtering (fast fail)"""
        
        # Add keyword checks (properly escape keywords)
        for keyword in keywords:
            # Escape single quotes in keywords to prevent syntax errors
            escaped_keyword = keyword.lower().replace("'", "\\'")
            prompt += f"\n    if '{escaped_keyword}' not in content_lower:"
            prompt += f"\n        return False"
        
        prompt += f"""
    # Regex verification (exact match)
    pattern = r'{regex_escaped}'
    return re.match(pattern, content) is not None

def parse_log_template_{template_id}(content: str) -> dict:
    \"\"\"Parse log and return template\"\"\"
    return {{
        'template_id': {template_id},
        'template': '{template_escaped}'
    }}
```

Generate ONLY the Python code above, no explanations:"""

        # Generate code
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.1,  # Low temp for deterministic code
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract code block if wrapped in markdown
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response.strip()
        
        return code
    
    def generate_batch(
        self,
        templates: List[dict],
        batch_size: int = 10
    ) -> List[str]:
        """
        Generate parser code for multiple templates in batches.
        
        Args:
            templates: List of template dicts with keys: id, template, keywords, pattern, count
            batch_size: Number of templates to process at once
        
        Returns:
            List of generated code strings
        """
        results = []
        
        for i in range(0, len(templates), batch_size):
            batch = templates[i:i+batch_size]
            logger.info(f"Generating batch {i//batch_size + 1}/{(len(templates)-1)//batch_size + 1}")
            
            for template_info in batch:
                code = self.generate_parser_code(
                    template_id=template_info['id'],
                    template=template_info['template'],
                    keywords=template_info['keywords'],
                    regex_pattern=template_info['pattern'],
                    count=template_info['count']
                )
                results.append(code)
        
        return results
    
    def unload(self):
        """Free GPU memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            logger.info("✅ Qwen model unloaded")
