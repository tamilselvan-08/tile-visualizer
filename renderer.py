import cv2
import numpy as np
from utils import logger, cache
from plane_detection import estimate_surface_quad

def warp_texture_perspective(texture_bgr, quad, out_shape):
    """
    Warps a texture image into the quadrilateral via homography.
    Uses sub-pixel interpolation for better quality.
    Improves Realism: Removes jagged stretching typical of basic warps.
    """
    out_h, out_w = out_shape
    th, tw = texture_bgr.shape[:2]
    
    src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    
    # Use INTER_CUBIC or INTER_LANCZOS4 for subpixel rendering quality
    warped = cv2.warpPerspective(
        texture_bgr, H, (out_w, out_h),
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped

def get_feathered_mask(mask_uint8, blur_radius=5):
    """
    Applies gradient alpha blending (feathering) to the mask edges.
    Improves Realism: Eliminates sharp, visible rendering borders, 
    making the tile blend naturally into adjacent walls or floors.
    """
    mask_f = cv2.GaussianBlur(mask_uint8, (blur_radius, blur_radius), sigmaX=0).astype(np.float32) / 255.0
    return mask_f

def render_surface(room_bgr, texture_bgr, mask_uint8, depth_map=None, K=None):
    """
    Perspective rendering step. 
    Warp the generated tiled canvas into the scene based on geometry.
    Returns the warped raw texture (before lighting).
    """
    h, w = room_bgr.shape[:2]
    quad = estimate_surface_quad(mask_uint8, depth_map, K)
    
    if quad is None:
        return None
        
    warped_texture = warp_texture_perspective(texture_bgr, quad, (h, w))
    return warped_texture
