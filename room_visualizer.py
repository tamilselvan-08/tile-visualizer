#!/usr/bin/env python3
"""
room_visualizer.py
-------------------
POC: automatically detect wall / floor / ceiling in a room photo and
apply a tile/texture image onto them with correct perspective and
preserved lighting/shadows (so it looks real, not pasted on).

Pipeline:
  1. Semantic segmentation (Mask2Former, trained on ADE20K) -> wall/floor/ceiling masks
  2. Mask cleanup (largest component, morphological smoothing, hole filling)
  3. Optional refinement with SAM (transformers) for pixel-accurate edges
  4. Quad-corner detection on each mask -> homography
  5. Texture tiling + perspective warp into that quad
  6. Lighting-preserving composite (LAB luminance blend) with feathered edges

Usage:
  python room_visualizer.py \
      --room demo-room.jpg \
      --wall-texture tiles/blue_marble.jpg \
      --floor-texture tiles/wood_floor.jpg \
      --output result.png

  # with SAM edge refinement (slower, sharper edges):
  python room_visualizer.py --room demo-room.jpg --wall-texture t.jpg --refine

  # save intermediate masks for debugging:
  python room_visualizer.py --room demo-room.jpg --wall-texture t.jpg --debug
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

from utils import logger, setup_logger
from segmentation import load_segmentation_model, segment_room, get_surface_mask





# --------------------------------------------------------------------------
# 3. Optional SAM refinement for pixel-accurate edges
# --------------------------------------------------------------------------
def refine_mask_with_sam(image_pil, mask_uint8, device="cuda",
                          sam_model_name="facebook/sam-vit-base"):
    """
    Uses a bounding box derived from the rough segmentation mask as a prompt
    for Segment Anything, producing a sharper, pixel-accurate mask.
    """
    try:
        import torch
        from transformers import SamModel, SamProcessor
    except ImportError:
        print("[warn] transformers SAM support not available, skipping --refine.")
        return mask_uint8

    ys, xs = np.where(mask_uint8 > 0)
    if len(xs) == 0:
        return mask_uint8
    box = [[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]]

    processor = SamProcessor.from_pretrained(sam_model_name)
    model = SamModel.from_pretrained(sam_model_name).to(device)

    inputs = processor(image_pil, input_boxes=[box], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]
    scores = outputs.iou_scores.cpu().numpy()[0, 0]
    best = masks[0][int(np.argmax(scores))].numpy().astype(np.uint8) * 255
    return best


# --------------------------------------------------------------------------
# 4. Quad-corner detection for the perspective homography
# --------------------------------------------------------------------------
def order_corners(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_surface_quad(mask_uint8):
    """
    Finds the 4 corners of the surface region for a perspective warp.
    Tries a 4-point polygon approximation first (handles walls at an angle
    correctly); falls back to a rotated bounding rect if the shape isn't
    quad-like (e.g. an irregular floor edge behind furniture).
    """
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        return order_corners(approx.reshape(4, 2))

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return order_corners(box)


# --------------------------------------------------------------------------
# 5. Texture tiling + perspective warp
# --------------------------------------------------------------------------
def tile_texture(texture_bgr, out_w, out_h, repeats_x=4, add_grout=True, grout_px=2):
    """
    Repeats the texture to cover a target canvas size (so a single tile
    photo doesn't get stretched into one giant blurry image), optionally
    drawing thin grout lines between repeats for a tile-like look.
    """
    th, tw = texture_bgr.shape[:2]
    tile_w = max(1, out_w // repeats_x)
    tile_h = int(tile_w * th / tw)

    tile = cv2.resize(texture_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

    if add_grout:
        tile = cv2.copyMakeBorder(
            tile, grout_px, grout_px, grout_px, grout_px,
            borderType=cv2.BORDER_CONSTANT, value=(190, 190, 190)
        )

    reps_x = int(np.ceil(out_w / tile.shape[1])) + 1
    reps_y = int(np.ceil(out_h / tile.shape[0])) + 1
    tiled = np.tile(tile, (reps_y, reps_x, 1))
    return tiled[:out_h, :out_w]


def warp_texture_to_quad(texture_bgr, quad, out_shape):
    """Warps a (tiled) texture image into the given quadrilateral via homography."""
    out_h, out_w = out_shape
from sam_refinement import load_sam_model, refine_mask_with_sam
from depth import load_depth_model, estimate_depth
from camera import get_camera_intrinsics
from texture_engine import generate_adaptive_grout_color, tile_texture_physically
from renderer import render_surface, get_feathered_mask
from lighting import extract_illumination, apply_illumination_transfer
from reflection import apply_reflections
from occlusion import restore_occlusions


# --------------------------------------------------------------------------
def process_surface(room_bgr, label_map, mask_uint8, texture_path, 
                    depth_map=None, K=None, add_grout=True, is_glossy=False):
    if mask_uint8.max() == 0:
        return room_bgr  # surface not detected in this photo

    texture_bgr = cv2.imread(texture_path)
    if texture_bgr is None:
        sys.exit(f"Could not read texture image: {texture_path}")

    h, w = room_bgr.shape[:2]
    
    # 1. Texture Engine
    grout_color = generate_adaptive_grout_color(room_bgr, mask_uint8) if add_grout else None
    grout_px = 2 if add_grout else 0
    tiled_texture = tile_texture_physically(texture_bgr, w, h, grout_px=grout_px, grout_color=grout_color)
    
    # 2. Perspective Engine
    warped_texture = render_surface(room_bgr, tiled_texture, mask_uint8, depth_map, K)
    if warped_texture is None:
        return room_bgr
        
    # 3. Lighting Engine
    L, ambient, detail = extract_illumination(room_bgr, mask_uint8)
    relit = apply_illumination_transfer(warped_texture, L, ambient, detail, mask_uint8)
    
    # 4. Reflection Engine
    relit = apply_reflections(relit, room_bgr, mask_uint8, is_glossy)
    
    # 5. Edge Engine (Blend with feathering)
    mask_f = get_feathered_mask(mask_uint8, blur_radius=5)
    mask_f_3d = np.repeat(mask_f[:, :, np.newaxis], 3, axis=2)
    
    blended = room_bgr.astype(np.float32) * (1 - mask_f_3d) + relit.astype(np.float32) * mask_f_3d
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # 6. Occlusion Engine
    final = restore_occlusions(room_bgr, blended, label_map)
    return final


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="AI room visualizer POC (wall/floor/ceiling retexturing)")
    ap.add_argument("--room", required=True, help="Path to room photo")
    ap.add_argument("--wall-texture", help="Texture/tile image to apply to the wall")
    ap.add_argument("--floor-texture", help="Texture/tile image to apply to the floor")
    ap.add_argument("--ceiling-texture", help="Texture/tile image to apply to the ceiling")
    ap.add_argument("--output", default="result.png", help="Output image path")
    ap.add_argument("--model", default="facebook/mask2former-swin-large-ade-semantic",
                     help="HF segmentation model (Mask2Former compatible)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--refine", action="store_true", help="Refine masks with SAM (sharper edges, slower)")
    ap.add_argument("--no-grout", action="store_true", help="Disable grout lines between tile repeats")
    ap.add_argument("--debug", action="store_true", help="Save intermediate masks next to output")
    args = ap.parse_args()

    if not any([args.wall_texture, args.floor_texture, args.ceiling_texture]):
        sys.exit("Provide at least one of --wall-texture / --floor-texture / --ceiling-texture")

    room_pil = Image.open(args.room).convert("RGB")
    room_bgr = cv2.cvtColor(np.array(room_pil), cv2.COLOR_RGB2BGR)

    logger.info(f"[1/4] Loading segmentation model ({args.model}) ...")
    processor, model, device = load_segmentation_model(args.model, args.device)

    logger.info("[2/4] Segmenting room (wall / floor / ceiling) ...")
    label_map = segment_room(room_pil, processor, model, device)

    logger.info("Loading SAM model...")
    sam_processor, sam_model, sam_device = load_sam_model(device=args.device) if args.refine else (None, None, None)
    
    logger.info("Loading Depth model...")
    depth_pipe, depth_device = load_depth_model(device=args.device)
    
    logger.info("Estimating depth and camera intrinsics...")
    depth_map = estimate_depth(room_pil, depth_pipe)
    K = get_camera_intrinsics(room_bgr.shape)

    surfaces = {
        "wall": args.wall_texture,
        "floor": args.floor_texture,
        "ceiling": args.ceiling_texture,
    }

    result = room_bgr.copy()
    for name, texture_path in surfaces.items():
        if texture_path is None:
            continue
        raw_mask = get_surface_mask(label_map, name, guide_bgr=room_bgr)

        if args.refine and raw_mask.max() > 0:
            logger.info(f"Refining {name} mask with SAM 2 ...")
            raw_mask = refine_mask_with_sam(room_pil, raw_mask, sam_processor, sam_model, sam_device)

        if args.debug:
            debug_path = os.path.splitext(args.output)[0] + f"_mask_{name}.png"
            cv2.imwrite(debug_path, raw_mask)
            logger.info(f"Saved debug mask: {debug_path}")

        logger.info(f"Applying texture to {name} ...")
        
        # Determine if tile is glossy (e.g. wall/floor marble). Currently standardizing for demo.
        is_glossy = True if "marble" in texture_path.lower() else False
        
        result = process_surface(
            result, label_map, raw_mask, texture_path, 
            depth_map=depth_map, K=K, add_grout=not args.no_grout, is_glossy=is_glossy
        )

    print(f"[4/4] Saving result to {args.output}")
    cv2.imwrite(args.output, result)
    print("Done.")


if __name__ == "__main__":
    main()