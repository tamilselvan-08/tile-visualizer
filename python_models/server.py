#!/usr/bin/env python3
"""
server.py
---------
Local API that the HTML tool (room_visualizer.html) calls to auto-detect
wall / floor / ceiling in an uploaded room photo, using the same
segmentation pipeline as room_visualizer.py.

Run it:
    python server.py

Then open room_visualizer.html in your browser and click "Auto-detect (AI)".
The HTML talks to this server at http://127.0.0.1:5000 — keep this terminal
window open while you use the tool.
"""

import base64
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

try:
    from flask_cors import CORS
except ImportError:
    sys.exit("Missing dependency. Run:\n  pip install flask flask-cors\n")

from python_models.room_visualizer import find_surface_quad
from python_models.segmentation import ADE20K_CLASSES, get_surface_mask, load_segmentation_model, segment_room, get_protected_mask
from python_models.plane_detection import detect_architectural_edges

app = Flask(__name__)
CORS(app)  # allow the HTML file (served from a different origin/port) to call this API

MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"  # fast enough for interactive use
_model_cache = {"processor": None, "model": None, "device": None}


def get_model():
    if _model_cache["model"] is None:
        print(f"Loading segmentation model ({MODEL_NAME}) — first request only, may take a minute...")
        processor, model, device = load_segmentation_model(MODEL_NAME, device="cuda")
        _model_cache.update(processor=processor, model=model, device=device)
        print("Model loaded.")
    return _model_cache["processor"], _model_cache["model"], _model_cache["device"]


def mask_to_base64_png(mask_uint8):
    ok, buf = cv2.imencode(".png", mask_uint8)
    return base64.b64encode(buf).decode("utf-8")


def estimate_plane_normal(name, quad):
    if name == "floor":
        return [0.0, 1.0, 0.0]
    elif name == "ceiling":
        return [0.0, -1.0, 0.0]
    elif "wall" in name or "pillar" in name:
        if len(quad) == 4:
            tl, tr, br, bl = quad
            h_left = np.linalg.norm(np.array(tl) - np.array(bl))
            h_right = np.linalg.norm(np.array(tr) - np.array(br))
            if h_left + h_right > 0:
                ratio = (h_left - h_right) / (h_left + h_right)
            else:
                ratio = 0.0
            theta = ratio * (np.pi / 3.0)  # max 60 degrees rotation
            nx = float(np.sin(theta))
            ny = 0.0
            nz = float(np.cos(theta))
            norm = np.sqrt(nx*nx + ny*ny + nz*nz)
            return [nx/norm, ny/norm, nz/norm]
    return [0.0, 0.0, 1.0]


def adjust_split_to_avoid_obstacles(x_split, protected_mask, max_search=120):
    h, w = protected_mask.shape
    best_x = x_split
    min_overlap = np.sum(protected_mask[:, x_split] > 0)
    if min_overlap == 0:
        return x_split
        
    for offset in range(1, max_search):
        for sign in [-1, 1]:
            test_x = x_split + sign * offset
            if 0 <= test_x < w:
                overlap = np.sum(protected_mask[:, test_x] > 0)
                if overlap == 0:
                    return test_x
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_x = test_x
    return best_x


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _model_cache["model"] is not None})


@app.route("/segment", methods=["POST"])
def segment():
    if "image" not in request.files:
        return jsonify({"error": "no 'image' file in request"}), 400

    try:
        img_pil = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400

    processor, model, device = get_model()
    label_map = segment_room(img_pil, processor, model, device)
    guide_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    protected_mask = get_protected_mask(label_map)

    scene_model = {
        "wall": [],
        "floor": [],
        "ceiling": []
    }
    
    # 1. Floor & Ceiling
    for name in ["floor", "ceiling"]:
        mask = get_surface_mask(label_map, name, guide_bgr=guide_bgr)
        if mask.max() == 0:
            continue
        quad = find_surface_quad(mask)
        if quad is None:
            continue
            
        plane_normal = estimate_plane_normal(name, quad.tolist())
        ys, xs = np.where(mask > 0)
        plane_bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else [0, 0, 0, 0]
        
        scene_model[name].append({
            "mask_png": mask_to_base64_png(mask),
            "quad": quad.tolist(),
            "width": mask.shape[1],
            "height": mask.shape[0],
            "plane_id": name,
            "plane_normal": plane_normal,
            "plane_bounds": plane_bounds
        })

    # 2. Wall planes with splitting
    wall_mask = get_surface_mask(label_map, "wall", guide_bgr=guide_bgr)
    adjusted_splits = []
    if wall_mask.max() > 0:
        splits = detect_architectural_edges(wall_mask, guide_bgr)
        adjusted_splits = [adjust_split_to_avoid_obstacles(s, protected_mask) for s in splits]
        adjusted_splits = sorted(list(set([s for s in adjusted_splits if s > 0 and s < wall_mask.shape[1]])))

        if len(adjusted_splits) == 0:
            # Single plane
            quad = find_surface_quad(wall_mask)
            if quad is not None:
                plane_normal = estimate_plane_normal("wall", quad.tolist())
                ys, xs = np.where(wall_mask > 0)
                plane_bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else [0, 0, 0, 0]
                scene_model["wall"].append({
                    "mask_png": mask_to_base64_png(wall_mask),
                    "quad": quad.tolist(),
                    "width": wall_mask.shape[1],
                    "height": wall_mask.shape[0],
                    "plane_id": "wall",
                    "plane_normal": plane_normal,
                    "plane_bounds": plane_bounds
                })
        else:
            w = wall_mask.shape[1]
            boundaries = [0] + adjusted_splits + [w]
            
            plane_names = []
            if len(adjusted_splits) == 1:
                plane_names = ["wall_left", "wall_right"]
            elif len(adjusted_splits) == 2:
                plane_names = ["wall_left", "wall_center", "wall_right"]
            else:
                plane_names = ["wall_left", "wall_center", "wall_right"] + [f"wall_{i}" for i in range(3, len(adjusted_splits) + 1)]
                
            for idx in range(len(boundaries) - 1):
                x_start = boundaries[idx]
                x_end = boundaries[idx+1]
                
                sub_mask = wall_mask.copy()
                if x_start > 0:
                    sub_mask[:, :x_start] = 0
                if x_end < w:
                    sub_mask[:, x_end:] = 0
                    
                if sub_mask.max() == 0:
                    continue
                quad = find_surface_quad(sub_mask)
                if quad is None:
                    continue
                    
                p_name = plane_names[idx] if idx < len(plane_names) else f"wall_{idx}"
                plane_normal = estimate_plane_normal(p_name, quad.tolist())
                ys, xs = np.where(sub_mask > 0)
                plane_bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else [0, 0, 0, 0]
                
                scene_model["wall"].append({
                    "mask_png": mask_to_base64_png(sub_mask),
                    "quad": quad.tolist(),
                    "width": sub_mask.shape[1],
                    "height": sub_mask.shape[0],
                    "plane_id": p_name,
                    "plane_normal": plane_normal,
                    "plane_bounds": plane_bounds
                })

    # 3. Columns/Pillars (ADE20K class 42)
    pillar_raw = label_map == 42
    if pillar_raw.any():
        from python_models.segmentation import clean_mask
        pillar_mask = clean_mask(pillar_raw, guide_bgr=guide_bgr)
        pillar_mask[protected_mask > 0] = 0
        
        if pillar_mask.max() > 0:
            p_splits = detect_architectural_edges(pillar_mask, guide_bgr)
            p_adjusted_splits = [adjust_split_to_avoid_obstacles(s, protected_mask) for s in p_splits]
            p_adjusted_splits = sorted(list(set([s for s in p_adjusted_splits if s > 0 and s < pillar_mask.shape[1]])))
            
            if len(p_adjusted_splits) == 0:
                quad = find_surface_quad(pillar_mask)
                if quad is not None:
                    plane_normal = estimate_plane_normal("pillar_front", quad.tolist())
                    ys, xs = np.where(pillar_mask > 0)
                    plane_bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else [0, 0, 0, 0]
                    scene_model["wall"].append({
                        "mask_png": mask_to_base64_png(pillar_mask),
                        "quad": quad.tolist(),
                        "width": pillar_mask.shape[1],
                        "height": pillar_mask.shape[0],
                        "plane_id": "pillar_front",
                        "plane_normal": plane_normal,
                        "plane_bounds": plane_bounds
                    })
            else:
                pw = pillar_mask.shape[1]
                p_boundaries = [0] + p_adjusted_splits + [pw]
                p_plane_names = []
                if len(p_adjusted_splits) == 1:
                    p_plane_names = ["pillar_left", "pillar_front"]
                else:
                    p_plane_names = ["pillar_left", "pillar_front", "pillar_right"] + [f"pillar_{i}" for i in range(3, len(p_adjusted_splits) + 1)]
                    
                for idx in range(len(p_boundaries) - 1):
                    px_start = p_boundaries[idx]
                    px_end = p_boundaries[idx+1]
                    
                    psub_mask = pillar_mask.copy()
                    if px_start > 0:
                        psub_mask[:, :px_start] = 0
                    if px_end < pw:
                        psub_mask[:, px_end:] = 0
                        
                    if psub_mask.max() == 0:
                        continue
                    quad = find_surface_quad(psub_mask)
                    if quad is None:
                        continue
                        
                    pp_name = p_plane_names[idx] if idx < len(p_plane_names) else f"pillar_{idx}"
                    plane_normal = estimate_plane_normal(pp_name, quad.tolist())
                    ys, xs = np.where(psub_mask > 0)
                    plane_bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else [0, 0, 0, 0]
                    
                    scene_model["wall"].append({
                        "mask_png": mask_to_base64_png(psub_mask),
                        "quad": quad.tolist(),
                        "width": psub_mask.shape[1],
                        "height": psub_mask.shape[0],
                        "plane_id": pp_name,
                        "plane_normal": plane_normal,
                        "plane_bounds": plane_bounds
                    })

    # Add metadata for debug lines/splits
    scene_model["_metadata"] = {
        "splits": adjusted_splits
    }

    # Verify if we detected anything
    has_surfaces = any(len(scene_model[cat]) > 0 for cat in ["wall", "floor", "ceiling"])
    if not has_surfaces:
        return jsonify({"error": "no wall/floor/ceiling detected in this photo"}), 200

    return jsonify(scene_model)


if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    print("Leave this window open, then use room_visualizer.html in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=True)