import cv2
import numpy as np
from geometry import line_intersection
from utils import logger, cache

def estimate_vanishing_point(image_bgr, mask_uint8):
    """
    Estimates the primary vanishing point of a surface (e.g. floor or wall)
    using line segment detection (LSD) and finding intersection points.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Only consider lines within the masked area or its bounds
    gray_masked = cv2.bitwise_and(gray, gray, mask=cv2.dilate(mask_uint8, np.ones((50,50), np.uint8)))
    
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(gray_masked)
    
    if lines is None or len(lines) < 2:
        return None
    
    # Filter lines by length
    valid_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length > 50:
            valid_lines.append(((x1, y1), (x2, y2)))
            
    if len(valid_lines) < 2:
        return None
        
    # Find intersections of strongest lines
    intersections = []
    for i in range(len(valid_lines)):
        for j in range(i+1, len(valid_lines)):
            pt = line_intersection(valid_lines[i], valid_lines[j])
            if pt:
                # Basic filter: reject intersections too far outside image
                h, w = gray.shape
                if -w < pt[0] < 2*w and -h < pt[1] < 2*h:
                    intersections.append(pt)
                    
    if not intersections:
        return None
        
    # Geometric median or mean of intersections as a proxy for VP
    intersections = np.array(intersections)
    vp = np.median(intersections, axis=0)
    return (int(vp[0]), int(vp[1]))

def estimate_focal_length(image_shape):
    """Fallback focal length estimation assuming standard FOV (e.g. 60 deg)."""
    h, w = image_shape[:2]
    fov = 60 * np.pi / 180
    focal_length = (max(h, w) / 2.0) / np.tan(fov / 2.0)
    return focal_length

def get_camera_intrinsics(image_shape, focal_length=None):
    h, w = image_shape[:2]
    if focal_length is None:
        focal_length = estimate_focal_length(image_shape)
        
    cx, cy = w / 2.0, h / 2.0
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K
