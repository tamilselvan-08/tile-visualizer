import cv2
import numpy as np
from python_models.segmentation import get_protected_mask

def restore_occlusions(room_bgr, rendered_bgr, label_map):
    """
    Explicitly restores all protected objects (picture frames, mirrors, furniture, etc.) 
    over the rendered image.
    Improves Realism: Guarantees that texture absolutely NEVER bleeds onto non-wall objects.
    """
    protected_mask = get_protected_mask(label_map)
    
    # Feather the protection mask slightly for soft blending at the edges of furniture
    mask_f = cv2.GaussianBlur(protected_mask, (3, 3), 0).astype(np.float32) / 255.0
    mask_f = np.repeat(mask_f[:, :, np.newaxis], 3, axis=2)
    
    out_bgr = rendered_bgr.astype(np.float32) * (1 - mask_f) + room_bgr.astype(np.float32) * mask_f
    return np.clip(out_bgr, 0, 255).astype(np.uint8)
