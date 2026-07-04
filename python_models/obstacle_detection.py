"""
obstacle_detection.py
----------------------
Classical-CV safety net for wall-mounted flat objects (picture frames,
mirrors, wall clocks, switch plates) that the AI segmentation model
sometimes fails to classify at all — meaning it labels those pixels
directly as "wall", so the object-protection layer (which only subtracts
pixels the AI DID correctly classify as furniture/decor) has nothing to
work with.

This doesn't rely on semantic understanding at all: most wall decor is
geometrically simple (a rectangle with a strong-contrast border, or a
circle for a clock/mirror), which classical edge detection finds reliably
regardless of what the AI model thinks the pixels are.
"""

import cv2
import numpy as np


def detect_frame_like_objects(guide_bgr, surface_mask_uint8,
                               min_area_ratio=0.003, max_area_ratio=0.25,
                               min_border_contrast=10, max_repeats=4):
    """
    Finds rectangular, frame-like objects sitting on top of a surface
    (wall/ceiling) using edge + contour analysis, independent of any AI
    classification. Returns a binary mask (0/255) of pixels that should be
    excluded from texturing.

    Tuned to be conservative:
      - requires a clear 4-corner rectangle with real contrast at its border
      - must sit mostly INSIDE the surface's own mask
      - CRITICAL: rejects any rectangle size/shape that repeats more than
        `max_repeats` times across the image. A real frame is a one-off;
        tile grout lines produce dozens of near-identical small rectangles
        across the whole wall, which would otherwise get misdetected as
        "frames" and exclude the entire tiled wall.
    """
    h, w = surface_mask_uint8.shape
    total_area = h * w
    min_area = min_area_ratio * total_area
    max_area = max_area_ratio * total_area

    gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 15, 60)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []  # (approx, area, aspect_bucket)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, bw, bh = cv2.boundingRect(approx)
        if bh == 0:
            continue
        aspect = bw / float(bh)
        if aspect < 0.3 or aspect > 3.5:
            continue

        roi = surface_mask_uint8[y:y + bh, x:x + bw]
        if roi.size == 0 or (roi > 0).mean() < 0.5:
            continue

        border_contrast = _edge_contrast(gray, approx)
        if border_contrast < min_border_contrast:
            continue

        candidates.append((approx, area, aspect))

    # repetition filter: bucket by (area, aspect) rounded to tolerance bands;
    # a bucket with many members is a repeating tile pattern, not a frame
    def bucket_key(area, aspect):
        return (round(area / (0.15 * area + 1e-6)), round(aspect / 0.15))

    from collections import Counter
    counts = Counter(bucket_key(a, r) for _, a, r in candidates)

    obstacles = np.zeros((h, w), dtype=np.uint8)
    for approx, area, aspect in candidates:
        if counts[bucket_key(area, aspect)] > max_repeats:
            continue  # repeats too often across the image -> tile grid, not a frame
        cv2.drawContours(obstacles, [approx], -1, 255, thickness=cv2.FILLED)

    return obstacles


def _edge_contrast(gray, quad):
    """Rough estimate of how strong the boundary contrast is, to reject weak/false edges."""
    mask_in = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_in, [quad], -1, 255, thickness=cv2.FILLED)
    mask_ring = cv2.dilate(mask_in, np.ones((9, 9), np.uint8)) - mask_in

    inner_vals = gray[mask_in > 0]
    ring_vals = gray[mask_ring > 0]
    if inner_vals.size == 0 or ring_vals.size == 0:
        return 0
    return abs(float(inner_vals.mean()) - float(ring_vals.mean()))