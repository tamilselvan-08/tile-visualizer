import cv2
import numpy as np

def extract_specular_highlights(room_bgr, mask_uint8, threshold=200):
    """
    Extracts the brightest reflections and window lights from the original wall.
    Improves Realism: A glossy tile must reflect the room's lights just like the old surface did.
    """
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find the brightest spots (specular highlights, window reflections)
    _, highlights = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    highlights = cv2.bitwise_and(highlights, highlights, mask=mask_uint8)
    
    # Smooth them to create a natural bloom/gloss map
    gloss_map = cv2.GaussianBlur(highlights, (21, 21), sigmaX=0).astype(np.float32) / 255.0
    return gloss_map

def apply_reflections(relit_bgr, room_bgr, mask_uint8, is_glossy=False):
    """
    Overlays the specular highlights back onto the rendered tile if it's glossy.
    """
    if not is_glossy:
        return relit_bgr
        
    gloss_map = extract_specular_highlights(room_bgr, mask_uint8)
    gloss_map_3d = np.repeat(gloss_map[:, :, np.newaxis], 3, axis=2)
    
    # Additive blending for specular highlights
    out_bgr = relit_bgr.astype(np.float32) + (room_bgr.astype(np.float32) * gloss_map_3d)
    return np.clip(out_bgr, 0, 255).astype(np.uint8)
