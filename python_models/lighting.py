import cv2
import numpy as np
from python_models.utils import logger, cache

def extract_illumination(room_bgr, mask_uint8):
    """
    Extracts multi-scale illumination components from the original room image.
    Separates:
      - Ambient low-frequency lighting (room gradients)
      - Shadows (high-frequency dark spots)
      - Highlights (high-frequency bright spots)
    Improves Realism: Preserves natural wall imperfections, window light, and micro-shadows.
    """
    # Convert to LAB for luminance extraction
    lab = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    
    # 1. Ambient Lighting (Base layer, blurred)
    ambient = cv2.GaussianBlur(L, (0, 0), sigmaX=31)
    
    # 2. Detail Layer (High frequency: shadows and highlights)
    detail = L - ambient
    
    return L, ambient, detail

def apply_illumination_transfer(warped_texture_bgr, original_l, ambient, detail, mask_uint8):
    """
    Transfers the extracted physical illumination to the flat warped texture.
    Avoids alpha blending which makes the texture look "transparent".
    Instead, it physically modulates the texture luminance.
    """
    tex_lab = cv2.cvtColor(warped_texture_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tex_l = tex_lab[:, :, 0]
    
    # Calculate average brightness of the texture to maintain its identity
    valid_tex = tex_l[mask_uint8 > 0]
    if len(valid_tex) == 0:
        return warped_texture_bgr
        
    tex_mean = np.mean(valid_tex)
    
    # Target ambient matching: shift the texture's mean to match the room's ambient
    ambient_mean = np.mean(ambient[mask_uint8 > 0])
    ambient_ratio = ambient / (ambient_mean + 1e-6)
    
    # Base lit texture
    lit_l = tex_l * ambient_ratio
    
    # Add back the high-frequency shadows and highlights perfectly
    # We scale the detail slightly based on the texture brightness 
    # (darker tiles show fewer highlights, brighter tiles show less deep shadows)
    lit_l = lit_l + detail * 1.2
    
    # Clip and reconstruct
    lit_l = np.clip(lit_l, 0, 255)
    tex_lab[:, :, 0] = lit_l
    
    relit_bgr = cv2.cvtColor(tex_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return relit_bgr
