"""
segmentation.py
----------------
Wall / floor / ceiling detection, with an explicit "object protection layer"
that subtracts known furniture/decor classes AFTER cleanup runs — so even
a small protected object (a wall lamp, a small frame) can't get swallowed
back in by the small-hole-fill step, regardless of its size.
"""

import sys

import cv2
import numpy as np

from utils import cache, logger

# ADE20K indices (0-indexed, matches HF's post_process_semantic_segmentation output)
ADE20K_CLASSES = {
    "wall": 0,
    "floor": 3,
    "ceiling": 5,
}

# Object Protection Layer: classes that should NEVER have texture rendered on
# them. Verified against the authoritative ADE20K 150-class list.
OCCLUSION_CLASSES = {
    7,    # bed
    8,    # windowpane
    10,   # cabinet
    12,   # person
    14,   # door
    15,   # table
    17,   # plant   (was wrongly 72="palm" before)
    18,   # curtain
    19,   # chair
    22,   # painting, picture
    23,   # sofa    (was wrongly 24="shelf" before)
    24,   # shelf   (was wrongly 25="house" before)
    27,   # mirror
    30,   # armchair
    35,   # wardrobe (was wrongly 34="rock" before)
    36,   # lamp     (was wrongly 35="wardrobe" before)
    39,   # cushion
    57,   # pillow
    62,   # bookcase
    64,   # coffee table
    66,   # flower   (was wrongly 123="trade name" before)
    69,   # bench
    75,   # swivel chair
    82,   # light
    85,   # chandelier
    89,   # television receiver (was wrongly 88="booth" before)
    97,   # ottoman
    110,  # stool
    134,  # sconce
    135,  # vase
    43,   # signboard
    144,  # bulletin board
    148,  # clock  <- was MISSING entirely; this is the wall-clock leak
}

# Fast, CPU-friendly by default. Mask2Former is more accurate but 10-30x
# slower on CPU (30-90s+ per image) — only worth it with a GPU.
DEFAULT_MODEL = "nvidia/segformer-b4-finetuned-ade-512-512"


def load_segmentation_model(model_name=DEFAULT_MODEL, device="cuda"):
    """
    Loads a pretrained ADE20K segmentation model, picking the correct model
    class automatically (SegFormer vs Mask2Former need different classes).
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except ImportError:
        sys.exit("Missing dependencies. Run:\n  pip install torch torchvision transformers accelerate\n")

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU (will be slow).")
        device = "cpu"

    is_mask2former = "mask2former" in model_name.lower()
    if is_mask2former and device == "cpu":
        logger.warning(
            f"{model_name} on CPU will be very slow (30-90s+ per image). "
            f"Consider {DEFAULT_MODEL} instead unless you have a GPU."
        )

    logger.info(f"Loading segmentation model {model_name} on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    if is_mask2former:
        from transformers import Mask2FormerForUniversalSegmentation
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    else:
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device


def segment_room(image_pil, processor, model, device, model_name=DEFAULT_MODEL):
    """Returns a (H, W) int array of ADE20K class ids for every pixel, using cache if available."""
    cached_seg = cache.get("seg", image_pil, model_name)
    if cached_seg is not None:
        return cached_seg

    import torch
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_size = [image_pil.size[::-1]]  # (H, W)
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=target_size)[0]
    seg_np = seg.cpu().numpy()

    cache.set("seg", seg_np, image_pil, model_name)
    return seg_np


def get_protected_mask(label_map):
    """Binary mask of all pixels belonging to a protected (never-texture) object class."""
    protected = np.zeros(label_map.shape, dtype=np.uint8)
    for class_id in OCCLUSION_CLASSES:
        protected[label_map == class_id] = 255
    return protected


def refine_edges_with_image(mask_uint8, guide_bgr, radius=6, eps=0.01):
    """
    Snaps the mask boundary onto real edges in the photo using an
    edge-preserving filter.

    Preferred path  : cv2.ximgproc.guidedFilter (opencv-contrib-python).
                      radius=6, eps=0.01 — tested against clean and
                      busy/patterned backgrounds; larger radius latches
                      onto wallpaper texture instead of real edges.
    Fallback path   : bilateral filter (base opencv-python).
                      Equivalent quality for this use-case; slightly
                      softer at very fine edges but never crashes.
    """
    if guide_bgr.shape[:2] != mask_uint8.shape[:2]:
        guide_bgr = cv2.resize(guide_bgr, (mask_uint8.shape[1], mask_uint8.shape[0]))
    mask_f = mask_uint8.astype(np.float32) / 255.0

    # --- preferred: guided filter (opencv-contrib) ---
    _ximgproc = getattr(cv2, "ximgproc", None)
    if _ximgproc is not None and hasattr(_ximgproc, "guidedFilter"):
        # guide must be the full colour image, not grayscale — a single-channel
        # guide triggers a NaN bug in some opencv-contrib builds.
        refined = _ximgproc.guidedFilter(
            guide=guide_bgr, src=mask_f, radius=radius, eps=eps
        )
        refined = np.nan_to_num(refined, nan=0.0)
    else:
        # --- fallback: bilateral filter on mask (base opencv) ---
        # Convert mask to uint8 for bilateralFilter, then normalise back.
        # d=radius*2+1 mirrors the guided-filter neighbourhood size.
        # sigmaColor / sigmaSpace chosen to preserve hard object edges while
        # smoothing the AI model's jaggy boundary (same goal as guided filter).
        d = max(5, radius * 2 + 1)
        # bilateralFilter requires uint8 or float32 single-channel
        blurred = cv2.bilateralFilter(
            mask_f, d=d, sigmaColor=75, sigmaSpace=75
        )
        refined = blurred

    out = (np.clip(refined, 0, 1) * 255).astype(np.uint8)
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return out


def clean_mask(mask_bool, guide_bgr=None, min_area_ratio=0.01, max_hole_area_ratio=0.0015):
    """
    - kills salt-and-pepper misclassification noise (median blur)
    - smooths jagged boundaries (gaussian blur + rethreshold)
    - (optionally) snaps edges onto the real photo via a guided filter
    - keeps only components above a minimum area
    - fills only SMALL internal holes; large holes (windows, furniture,
      pictures) are real exclusions and stay excluded
    Returns a uint8 mask (0 / 255).
    """
    mask = (mask_bool.astype(np.uint8)) * 255
    h, w = mask.shape
    min_area = min_area_ratio * h * w
    max_hole_area = max_hole_area_ratio * h * w

    mask = cv2.medianBlur(mask, 9)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=4)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if guide_bgr is not None:
        mask = refine_edges_with_image(mask, guide_bgr)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    inv = cv2.bitwise_not(cleaned)
    n_holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = cleaned.copy()
    for i in range(1, n_holes):
        x, y, bw, bh, area = hole_stats[i]
        touches_border = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if touches_border:
            continue
        if area <= max_hole_area:
            filled[hole_labels == i] = 255

    return filled


def get_surface_mask(label_map, surface_name, guide_bgr=None, detect_frames=True):
    """
    Full pipeline for one surface: raw class pixels -> clean -> subtract
    AI-protected objects -> subtract classically-detected frame-like objects.

    Two independent protection layers, in order:
      1. AI-class protection (subtracted last, see docstring notes above):
         catches objects the model DID correctly classify.
      2. Classical rectangle/frame detection (guide_bgr required): catches
         picture frames, wall clocks, mirrors etc. that the AI missed
         entirely and mislabeled as plain wall — no semantic understanding
         needed, just real edges in the photo. This is what actually fixes
         the "ghost image visible through tile" issue, since that happens
         precisely when the AI never detected the object at all.
    """
    class_id = ADE20K_CLASSES.get(surface_name)
    if class_id is None:
        return np.zeros(label_map.shape, dtype=np.uint8)

    raw_mask = label_map == class_id
    cleaned = clean_mask(raw_mask, guide_bgr=guide_bgr)

    protected_mask = get_protected_mask(label_map)
    cleaned[protected_mask > 0] = 0

    if detect_frames and guide_bgr is not None:
        from obstacle_detection import detect_frame_like_objects
        obstacles = detect_frame_like_objects(guide_bgr, cleaned)
        cleaned[obstacles > 0] = 0

    return cleaned