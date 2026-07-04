import sys
import numpy as np
from PIL import Image
from utils import logger, cache

def load_depth_model(model_name="depth-anything/Depth-Anything-V2-Base-hf", device="cuda"):
    """Loads Depth Anything V2 for dense depth estimation."""
    try:
        import torch
        from transformers import pipeline
    except ImportError:
        sys.exit(
            "Missing dependencies. Run:\n"
            "  pip install torch torchvision transformers\n"
        )
    
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU for depth estimation.")
        device = "cpu"
    
    logger.info(f"Loading depth model {model_name} on {device}...")
    # Use pipeline for easier handling of depth-anything outputs
    depth_pipe = pipeline(task="depth-estimation", model=model_name, device=0 if device == "cuda" else -1)
    return depth_pipe, device

def estimate_depth(image_pil, depth_pipe):
    """
    Estimates a dense depth map for the image.
    Returns depth as a normalized float32 numpy array (0.0 to 1.0).
    """
    cached_depth = cache.get("depth_v2", image_pil)
    if cached_depth is not None:
        return cached_depth

    logger.info("Estimating depth map...")
    result = depth_pipe(image_pil)
    
    # Extract PIL Image and convert to numpy
    depth_image = result["depth"]
    depth_np = np.array(depth_image).astype(np.float32)
    
    # Normalize to 0-1
    if depth_np.max() > 0:
        depth_np = depth_np / depth_np.max()
        
    cache.set("depth_v2", depth_np, image_pil)
    return depth_np
