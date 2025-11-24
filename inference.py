from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    """
    Move model inputs to specified device (CPU/GPU/MPS).
    
    Args:
        model_inputs (dict): Dictionary containing model inputs (tensors)
        device (str): Target device ('cpu', 'cuda', or 'mps')
        
    Returns:
        dict: Model inputs with all tensors moved to the specified device
    """
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    """
    Preprocess image and text inputs for the PaliGemma model.
    
    Args:
        processor (PaliGemmaProcessor): Processor for handling image and text preprocessing
        prompt (str): Text prompt for the model
        image_file_path (str): Path to the input image file
        device (str): Target device for the tensors
        
    Returns:
        dict: Processed model inputs including tokenized text and image tensors
    """
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def _sample_top_p(probs: torch.Tensor, p: float):
    """
    Perform top-p (nucleus) sampling on probability distribution.
    
    Args:
        probs (torch.Tensor): Probability distribution over vocabulary
        p (float): Probability threshold for top-p sampling
        
    Returns:
        torch.Tensor: Sampled token index
    """
    # Sort probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # Calculate cumulative probabilities
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Create mask for probabilities below the threshold
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p
    # Zero out probabilities below threshold
    probs_sort[mask] = 0.0
    # Renormalize probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample from the filtered distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Map back to original vocabulary indices
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    """
    Run inference with PaliGemma model on an image and text prompt.
    
    Args:
        model (PaliGemmaForConditionalGeneration): Loaded PaliGemma model
        processor (PaliGemmaProcessor): Processor for input preprocessing
        device (str): Device to run inference on
        prompt (str): Text prompt for the model
        image_file_path (str): Path to input image file
        max_tokens_to_generate (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (higher = more random)
        top_p (float): Top-p sampling parameter
        do_sample (bool): Whether to use sampling instead of greedy decoding
    """
    # Preprocess inputs
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    # Initialize KV cache for efficient generation
    kv_cache = KVCache()

    # Generate tokens until stop token or max length
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # Run model forward pass
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        
        # Sample next token
        if do_sample:
            # Apply temperature scaling
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            # Greedy decoding - take most likely token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)
        
        # Stop if end-of-sequence token is generated
        if next_token.item() == stop_token:
            break
            
        # Update inputs for next iteration
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    # Decode generated tokens to text
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(prompt + decoded)


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    """
    Main function to run PaliGemma inference.
    
    Args:
        model_path (str): Path to the pretrained PaliGemma model
        prompt (str): Text prompt for the model
        image_file_path (str): Path to input image file
        max_tokens_to_generate (int): Maximum number of tokens to generate (default: 100)
        temperature (float): Sampling temperature (default: 0.8)
        top_p (float): Top-p sampling parameter (default: 0.9)
        do_sample (bool): Whether to use sampling (default: False - greedy decoding)
        only_cpu (bool): Force CPU usage even if GPU is available (default: False)
    """
    # Determine device
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    # Load model and tokenizer
    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    # Initialize processor with model configuration
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Run inference
    print("Running inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)