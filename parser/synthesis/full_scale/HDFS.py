"""
HDFS Trainer Module
===================
Offline training pipeline for HDFS (Hadoop Distributed File System) dataset.

This follows the same structure as BGL.py but adapted for HDFS log format.
"""

from .BGL import BGLTrainer


class HDFSTrainer(BGLTrainer):
    """
    HDFS Training Pipeline - Inherits from BGLTrainer.
    
    HDFS-specific adaptations:
    - Different log format parsing
    - Different metadata extraction
    - Same core pipeline (Discovery -> Vector Bank -> Generation)
    
    Usage:
        trainer = HDFSTrainer(
            drain_output_path="path/to/HDFS_drain_output.csv",
            output_path="training/HDFS_parser_bank.py"
        )
        trainer.run_pipeline()
    """
    
    def __init__(self, drain_output_path: str, output_path: str = "training/HDFS_parser_bank.py", use_llm: bool = False):
        super().__init__(drain_output_path, output_path, use_llm)
    
    def _generate_dispatcher(self) -> str:
        """Override to handle HDFS log format"""
        code = '''def process_log(raw_line: str) -> dict:
    """Main dispatcher for HDFS logs
    
    HDFS Format: "081109 203518 14 INFO dfs.DataNode$PacketResponder: ..."
    """
    # Extract content from HDFS format
    parts = raw_line.split(maxsplit=4)
    content = parts[4].strip() if len(parts) >= 5 else raw_line.strip()
    
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
    """Parse a batch/window of HDFS logs for Log-GraphSeqNet"""
    return [process_log(log)['template'] for log in logs]
'''
        
        return code


def train_hdfs(drain_output_path: str, output_path: str = "training/HDFS_parser_bank.py", use_llm: bool = False):
    """Convenience function to train HDFS parser bank"""
    trainer = HDFSTrainer(drain_output_path=drain_output_path, output_path=output_path, use_llm=use_llm)
    trainer.run_pipeline()
