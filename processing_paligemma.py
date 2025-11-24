from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# Standard normalization parameters for ImageNet
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    Format prompt by adding image tokens, BOS token, and newline as per PaliGemma training.
    
    Args:
        prefix_prompt (str): Original text prompt
        bos_token (str): Beginning-of-sequence token
        image_seq_len (int): Number of image tokens to prepend
        image_token (str): Special image token string
        
    Returns:
        str: Formatted prompt with image tokens, BOS token, and newline
    """
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Rescale image pixel values by a scale factor.
    
    Args:
        image (np.ndarray): Input image array
        scale (float): Scaling factor to multiply pixel values
        dtype (np.dtype): Target data type for output
        
    Returns:
        np.ndarray: Rescaled image array
    """
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    """
    Resize PIL image to target dimensions.
    
    Args:
        image (Image): Input PIL image
        size (Tuple[int, int]): Target size as (height, width)
        resample (Image.Resampling): Resampling method for resizing
        reducing_gap (int, optional): Optimization for resampling
        
    Returns:
        np.ndarray: Resized image as numpy array
    """
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """
    Normalize image with mean and standard deviation.
    
    Args:
        image (np.ndarray): Input image array
        mean (Union[float, Iterable[float]]): Mean values for normalization
        std (Union[float, Iterable[float]]): Standard deviation values for normalization
        
    Returns:
        np.ndarray: Normalized image array
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Preprocess list of images for model input.
    
    Applies resizing, rescaling, normalization, and channel reordering.
    
    Args:
        images (List[Image.Image]): List of input PIL images
        size (Dict[str, int]): Target size dictionary
        resample (Image.Resampling): Resampling method for resize
        rescale_factor (float): Factor to rescale pixel values (typically 1/255)
        image_mean (Optional[Union[float, List[float]]]): Mean for normalization
        image_std (Optional[Union[float, List[float]]]): Std for normalization
        
    Returns:
        List[np.ndarray]: List of preprocessed image arrays in CHW format
    """
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:
    """
    Processor for PaliGemma multimodal inputs.
    
    Handles image preprocessing and text tokenization with special tokens for
    images, bounding boxes, and segmentation masks.
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        Initialize PaliGemma processor.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            num_image_tokens (int): Number of image tokens to use in prompt
            image_size (int): Target image size for resizing
        """
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        Process images and text into model inputs.
        
        Args:
            text (List[str]): List of text prompts (currently supports single item)
            images (List[Image.Image]): List of input images (currently supports single item)
            padding (str): Padding strategy for tokenizer
            truncation (bool): Whether to truncate sequences
            
        Returns:
            dict: Dictionary containing:
                - pixel_values: Preprocessed image tensors
                - input_ids: Tokenized text inputs
                - attention_mask: Attention masks for text inputs
                
        Note: Currently supports batch size of 1 for both images and text
        """
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # Preprocess images: resize, normalize, and convert to tensor
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data