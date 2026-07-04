"""
plane_detection.py
-------------------
3D plane fitting utilities. `estimate_surface_quad` currently uses only the
2D contour method (fast, no depth model needed) — the depth+RANSAC path
below is implemented and tested standalone, but intentionally NOT wired in
yet, since it needs a depth estimation pass per image which adds real
latency on CPU. Wire it in via `fit_plane_ransac` + `depth_to_point_cloud`
when there's a concrete case the 2D method fails on (e.g. a strongly angled
wall) and a GPU to run it on.
"""

import cv2
import numpy as np

from geometry import order_corners
from utils import logger


def depth_to_point_cloud(depth_map, K):
    """
    Unprojects a dense depth map to a 3D point cloud (H, W, 3) using camera
    intrinsics K. Not currently called by estimate_surface_quad — see
    module docstring.
    """
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth_map
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.dstack((X, Y, Z))


def fit_plane_ransac(points_3d, threshold=0.01, max_iters=100):
    """
    Fits a plane (a, b, c, d) to a set of 3D points using RANSAC. Not
    currently called by estimate_surface_quad — see module docstring.
    """
    best_plane = None
    best_inliers = 0
    n_points = points_3d.shape[0]

    if n_points < 3:
        return None

    for _ in range(max_iters):
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points_3d[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue

        normal = normal / norm
        d = -np.dot(normal, p1)

        distances = np.abs(np.dot(points_3d, normal) + d)
        inliers = np.sum(distances < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal[0], normal[1], normal[2], d)

    return best_plane


def estimate_surface_quad(mask_uint8, depth_map=None, K=None):
    """
    Finds the 4 corners of the surface region for a perspective warp, using
    2D contour approximation (fast, works well for a mostly-flat surface
    facing the camera). Tries a 4-point polygon approximation first; falls
    back to a rotated bounding rect if the shape isn't quad-like (e.g. an
    irregular floor edge behind furniture).

    `depth_map` and `K` are accepted for call-site compatibility but are
    CURRENTLY IGNORED — the 3D plane-fitting path (fit_plane_ransac +
    depth_to_point_cloud, above) isn't wired in yet. See module docstring.
    """
    if depth_map is not None:
        logger.debug("estimate_surface_quad: depth_map provided but not yet used (2D-only path active)")

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
