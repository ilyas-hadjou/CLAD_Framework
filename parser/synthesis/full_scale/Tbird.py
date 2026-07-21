"""
Thunderbird Trainer Module
===========================
Offline training pipeline for Thunderbird supercomputer dataset.

This follows the same structure as BGL.py but adapted for Thunderbird log format.
"""

from .BGL import BGLTrainer


class TbirdTrainer(BGLTrainer):
    """
    Thunderbird Training Pipeline - Inherits from BGLTrainer.
    
    Thunderbird-specific adaptations:
    - Different log format parsing
    - Different metadata extraction
    - Same core pipeline (Discovery -> Vector Bank -> Generation)
    
    Usage:
        trainer = TbirdTrainer(
            drain_output_path="path/to/Thunderbird_drain_output.csv",
            output_path="training/Tbird_parser_bank.py"
        )
        trainer.run_pipeline()
    """
    
    def __init__(self, drain_output_path: str, output_path: str = "training/Tbird_parser_bank.py", use_llm: bool = False):
        super().__init__(drain_output_path, output_path, use_llm)
    
    def _generate_dispatcher(self) -> str:
        """Override to handle Thunderbird log format"""
        code = '''def process_log(raw_line: str) -> dict:
    """Main dispatcher for Thunderbird logs
    
    Thunderbird Format: "- 1130213920 2005.10.25 node-12 kernel: ..."
    """
    # Extract content from Thunderbird format (similar to BGL)
    parts = raw_line.split(maxsplit=5)
    content = parts[5].strip() if len(parts) >= 6 else raw_line.strip()
    
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
    """Parse a batch/window of Thunderbird logs for Log-GraphSeqNet"""
    return [process_log(log)['template'] for log in logs]
'''
        
        return code


def train_tbird(drain_output_path: str, output_path: str = "training/Tbird_parser_bank.py", use_llm: bool = False):
    """Convenience function to train Thunderbird parser bank"""
    trainer = TbirdTrainer(drain_output_path=drain_output_path, output_path=output_path, use_llm=use_llm)
    trainer.run_pipeline()
