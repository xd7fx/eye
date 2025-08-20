import io
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# =========================
# App Config
# =========================
st.set_page_config(page_title="PeduliOpticv2 — Pupil Detector", layout="wide")
st.title("PeduliOpticv2 — Pupil Detector")

# API Key and Model ID
API_KEY = st.secrets.get("ROBOFLOW_API_KEY", "")
MODEL_ID = "peduliopticv2/4"

# =========================
# Drawing function
# =========================
def draw_pupil(bgr, preds):
    im = bgr.copy()
    for p in preds:
        x, y, w, h = p.get("x"), p.get("y"), p.get("width"), p.get("height")
        conf = p.get("confidence", 0)

        # Force label to "pupil"
        cls = "pupil"

        # Convert to box coordinates
        x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 210, 0), 2)

        # Label text
        lbl = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(im, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 210, 0), -1)
        cv2.putText(im, lbl, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return im

# =========================
# Inference function
# =========================
def infer_bytes(img_bytes):
    url = f"https://detect.roboflow.com/{MODEL_ID}"
    params = {"api_key": API_KEY, "confidence": 0.4, "overlap": 0.5}
    files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
    r = requests.post(url, params=params, files=files, timeout=30)
    r.raise_for_status()
    return r.json()

# =========================
# UI
# =========================
mode = st.radio("Choose input:", ["📷 Take a picture", "📁 Upload an image"])

if mode == "📷 Take a picture":
    cam = st.camera_input("Capture from webcam")
    if cam and st.button("Run detection"):
        pil = Image.open(cam).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        data = infer_bytes(buf.getvalue())
        preds = data.get("predictions", [])
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        vis = draw_pupil(bgr, preds)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                 channels="RGB", use_column_width=True)
        st.success(f"Detections: {len(preds)}")

elif mode == "📁 Upload an image":
    up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
    if up and st.button("Run detection"):
        pil = Image.open(up).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        data = infer_bytes(buf.getvalue())
        preds = data.get("predictions", [])
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        vis = draw_pupil(bgr, preds)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                 channels="RGB", use_column_width=True)
        st.success(f"Detections: {len(preds)}")
