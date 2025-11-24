from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Load PaliGemma model and tokenizer from HuggingFace format.
    
    This function handles loading the model configuration, tokenizer, and model weights
    from safetensors files, then initializes and returns the complete model.
    
    Args:
        model_path (str): Path to the directory containing model files (config.json, *.safetensors)
        device (str): Target device for model loading ('cpu', 'cuda', 'mps')
        
    Returns:
        Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]: 
            - Loaded PaliGemma model instance
            - Configured tokenizer instance
            
    Raises:
        AssertionError: If tokenizer padding side is not set to 'right'
        FileNotFoundError: If required model files are missing
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    # Ensure tokenizer uses right padding for autoregressive generation
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files in the model directory
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        # Use safetensors library to safely load tensor files
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config from config.json
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        # Create PaliGemmaConfig from loaded JSON
        config = PaliGemmaConfig(**model_config_file)

    # Create the model instance using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model from the collected tensors
    # strict=False allows for missing keys (e.g., if some weights are tied)
    model.load_state_dict(tensors, strict=False)

    # Tie weights between input and output embeddings for efficiency
    model.tie_weights()

    return (model, tokenizer)