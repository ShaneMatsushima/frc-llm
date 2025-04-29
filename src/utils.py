import json
import yaml
from pathlib import Path
import logging
from typing import Dict, Any, List, Union
import numpy as np
import torch

def setup_logging(log_file: str = 'frc_llm.log') -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Set random seed to {seed}")

def format_frc_rules(rules_dict: Dict[str, Any]) -> str:
    """Format FRC rules dictionary into a readable string"""
    formatted = []
    for section, content in rules_dict.items():
        formatted.append(f"=== {section.upper()} ===")
        if isinstance(content, dict):
            for rule_num, rule_text in content.items():
                formatted.append(f"{rule_num}: {rule_text}")
        elif isinstance(content, list):
            formatted.extend(content)
        formatted.append("")
    return "\n".join(formatted)

def split_data(data: List[Any], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Any]]:
    """Split data into train, validation, and test sets"""
    assert train_ratio + val_ratio < 1.0, "Sum of ratios must be less than 1"
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }

def save_model_artifacts(model, tokenizer, output_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """Save model, tokenizer, and config to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Saved model artifacts to {output_dir}")

def load_model_artifacts(model_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load model artifacts from disk"""
    model_dir = Path(model_dir)
    artifacts = {
        'model_dir': model_dir,
        'config_path': model_dir / "training_config.json"
    }
    
    with open(artifacts['config_path'], 'r') as f:
        artifacts['config'] = json.load(f)
    
    return artifacts

def calculate_bleu(references: List[List[str]], hypotheses: List[str]) -> float:
    """Calculate BLEU score for evaluation"""
    from nltk.translate.bleu_score import corpus_bleu
    # Tokenize if not already tokenized
    refs = [[ref.split() for ref in ref_group] for ref_group in references]
    hyps = [hyp.split() for hyp in hypotheses]
    return corpus_bleu(refs, hyps)

def get_device() -> torch.device:
    """Get the appropriate device (GPU if available)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")