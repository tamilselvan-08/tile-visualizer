import numpy as np
import sys
from python_models.utils import logger, cache

def load_sam_model(model_name="facebook/sam-vit-large", device="cuda"):
    """
    Loads a Segment Anything model for pixel-perfect mask refinement.
    Defaults to SAM large for high quality.
    """
    try:
        import torch
        from transformers import SamModel, SamProcessor
    except ImportError:
        logger.error("transformers not installed.")
        return None, None, device

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available for SAM, falling back to CPU.")
        device = "cpu"

    logger.info(f"Loading SAM model {model_name} on {device}...")
    try:
        processor = SamProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name).to(device)
    except Exception as e:
        logger.error(f"Failed to load SAM model {model_name}: {e}")
        return None, None, device
    
    return processor, model, device

def refine_mask_with_sam(image_pil, mask_uint8, processor, model, device):
    """
    Refines a rough mask (e.g. from Mask2Former) using SAM.
    Uses the bounding box of the rough mask as a prompt.
    """
    if model is None or processor is None:
        logger.warning("SAM not loaded. Returning original mask.")
        return mask_uint8

    # The prompt will be the bounding box of the largest component to avoid stray pixels
    import cv2
    import torch
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_uint8
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    box = [[x, y, x + w, y + h]]
    
    # Check cache
    cache_key = f"sam_refined_{x}_{y}_{w}_{h}"
    cached = cache.get(cache_key, image_pil)
    if cached is not None:
        return cached

    logger.info("Refining mask with SAM...")
    inputs = processor(image_pil, input_boxes=[box], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]
    
    scores = outputs.iou_scores.cpu().numpy()[0, 0]
    best_idx = int(np.argmax(scores))
    best_mask = masks[0][best_idx].numpy().astype(np.uint8) * 255
    
    # Combine original constraint with SAM's sharp edges to prevent bleeding
    refined = cv2.bitwise_and(best_mask, cv2.dilate(mask_uint8, np.ones((15, 15), np.uint8)))
    
    cache.set(cache_key, refined, image_pil)
    return refined
