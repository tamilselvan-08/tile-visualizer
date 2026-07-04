import cv2
import numpy as np
import random
from python_models.utils import logger, cache

def apply_random_variation(tile_bgr):
    """
    Applies micro color variations, slight brightness adjustments,
    and potentially flips to avoid obvious repeating patterns.
    Improves Realism: Prevents the eye from catching "cloned" tiles,
    a common artifact in synthetic rendering.
    """
    h, w = tile_bgr.shape[:2]
    out = tile_bgr.copy().astype(np.float32)
    
    # Micro brightness variation (+/- 5%)
    brightness_factor = random.uniform(0.95, 1.05)
    out = out * brightness_factor
    
    # Micro color variation (slight tint shifts)
    tint = np.array([random.uniform(0.98, 1.02) for _ in range(3)], dtype=np.float32)
    out = out * tint
    
    # Random flips
    if random.choice([True, False]):
        out = cv2.flip(out, 1) # horizontal
    if random.choice([True, False]):
        out = cv2.flip(out, 0) # vertical
        
    return np.clip(out, 0, 255).astype(np.uint8)

def generate_adaptive_grout_color(room_bgr, mask_uint8, custom_grout_color=None):
    """
    Automatically adapts grout color based on room brightness and shadow intensity.
    Improves Realism: Prevents glowing/unrealistic bright lines in dark rooms.
    """
    if custom_grout_color is not None:
        return custom_grout_color
        
    # Find average brightness of the targeted surface
    masked_room = cv2.bitwise_and(room_bgr, room_bgr, mask=mask_uint8)
    hsv = cv2.cvtColor(masked_room, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    valid_pixels = v_channel[mask_uint8 > 0]
    if len(valid_pixels) == 0:
        return (100, 100, 100) # Fallback gray
        
    median_v = np.median(valid_pixels)
    
    # Darker rooms get darker grout. 
    # Usually grout is slightly darker and less saturated than the average ambient.
    grout_v = max(30, int(median_v * 0.6))
    return (grout_v, grout_v, grout_v) # BGR gray

def tile_texture_physically(texture_bgr, out_w, out_h, repeats_x=4, 
                            grout_px=2, grout_color=(100,100,100)):
    """
    Generates a large tiled canvas.
    Applies variations per tile.
    Improves Realism: Removes the "flat sticker" look by treating each tile as an individual object.
    """
    th, tw = texture_bgr.shape[:2]
    tile_w = max(1, out_w // repeats_x)
    tile_h = int(tile_w * th / tw)
    
    base_tile = cv2.resize(texture_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    
    reps_x = int(np.ceil(out_w / (tile_w + grout_px))) + 1
    reps_y = int(np.ceil(out_h / (tile_h + grout_px))) + 1
    
    canvas = np.full((out_h, out_w, 3), grout_color, dtype=np.uint8)
    
    # Soft grout edges preparation
    # By placing tiles into the canvas, the remaining background is the grout.
    # To make grout soft, we can later apply a slight blur to the canvas before perspective warp.
    
    for y in range(reps_y):
        for x in range(reps_x):
            start_y = y * (tile_h + grout_px)
            start_x = x * (tile_w + grout_px)
            
            if start_y >= out_h or start_x >= out_w:
                continue
                
            end_y = min(start_y + tile_h, out_h)
            end_x = min(start_x + tile_w, out_w)
            
            # Apply variation per tile instance
            varied_tile = apply_random_variation(base_tile)
            
            # Calculate slicing if at the edge
            slice_h = end_y - start_y
            slice_w = end_x - start_x
            
            canvas[start_y:end_y, start_x:end_x] = varied_tile[:slice_h, :slice_w]
            
    # Soften grout lines
    if grout_px > 0:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
            
    return canvas
