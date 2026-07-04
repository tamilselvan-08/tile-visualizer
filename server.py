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

from room_visualizer import find_surface_quad
from segmentation import ADE20K_CLASSES, get_surface_mask, load_segmentation_model, segment_room

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

    result = {}
    for name in ADE20K_CLASSES:
        mask = get_surface_mask(label_map, name, guide_bgr=guide_bgr)
        if mask.max() == 0:
            continue
        quad = find_surface_quad(mask)
        if quad is None:
            continue
        result[name] = {
            "mask_png": mask_to_base64_png(mask),
            "quad": quad.tolist(),  # [[x,y],[x,y],[x,y],[x,y]] in tl,tr,br,bl order
            "width": mask.shape[1],
            "height": mask.shape[0],
        }

    if not result:
        return jsonify({"error": "no wall/floor/ceiling detected in this photo"}), 200

    return jsonify(result)


if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    print("Leave this window open, then use room_visualizer.html in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)