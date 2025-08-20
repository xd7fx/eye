import os, io, time, json, cv2, numpy as np, requests, streamlit as st
from PIL import Image

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="PeduliOpticv2 Online", layout="wide")
st.title("PeduliOpticv2 — Online (Webcam / Phone)")

st.sidebar.header("Roboflow")
api_key = st.sidebar.text_input("API Key", value=st.secrets.get("ROBOFLOW_API_KEY", ""), type="password")
model_id = st.sidebar.text_input("Model ID", value="peduliopticv2/4")

st.sidebar.header("Params")
confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.4, 0.01)
overlap = st.sidebar.slider("Overlap (IoU)", 0.0, 1.0, 0.5, 0.01)
every_n_frames = st.sidebar.slider("تحليل كل كم فريم؟", 1, 10, 3) 

mode = st.radio("الوضع", ["Realtime (Webcam in Browser)", "Single Image"])

left, right = st.columns([2, 1])
img_ph = left.empty()
json_ph = right.empty()
status_ph = st.empty()

def draw_dets(bgr, preds):
    im = bgr.copy()
    for p in preds:
        x, y, w, h = p.get("x"), p.get("y"), p.get("width"), p.get("height")
        cls = p.get("class", "obj")
        conf = p.get("confidence", 0)
        x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 210, 0), 2)
        lbl = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(im, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 210, 0), -1)
        cv2.putText(im, lbl, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cx, cy = int(x), int(y)
        cv2.circle(im, (cx, cy), 3, (0, 210, 0), -1)
    return im

def infer_bytes(img_bytes):
    if not api_key or not model_id:
        raise RuntimeError("API key أو model_id مفقود.")
    url = f"https://detect.roboflow.com/{model_id}"
    params = {"api_key": api_key, "confidence": confidence, "overlap": overlap}
    files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
    r = requests.post(url, params=params, files=files, timeout=30)
    r.raise_for_status()
    return r.json()

if mode.startswith("Realtime"):
    st.caption("يشتغل عبر HTTPS ويطلب إذن الكاميرا من المتصفح.")
    rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    frame_count = {"n": 0}

    class VideoProcessor:
        def __init__(self):
            self.last_json = None
            self.last_time = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            frame_count["n"] += 1

            if frame_count["n"] % every_n_frames == 0:
                try:
                    ok, buf = cv2.imencode(".jpg", img)
                    if ok:
                        data = infer_bytes(buf.tobytes())
                        preds = data.get("predictions", [])
                        vis = draw_dets(img, preds)
                        self.last_json = data
                        img_out = vis
                        json_ph.json(self.last_json)
                        status_ph.info(f"Detections: {len(preds)}")
                    else:
                        img_out = img
                except Exception as e:
                    status_ph.error(f"Error: {e}")
                    img_out = img
            else:
                img_out = img

            return av.VideoFrame.from_ndarray(img_out, format="bgr24")

    webrtc_streamer(
        key="pupil-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

else:
    submode = st.radio("المصدر", ["Camera (Browser)", "Upload Image", "Image URL"])

    if submode == "Camera (Browser)":
        cam = st.camera_input("التقط صورة")
        if cam and st.button("Run Detection"):
            pil = Image.open(cam).convert("RGB")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=90)
            try:
                data = infer_bytes(buf.getvalue())
                preds = data.get("predictions", [])
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                vis = draw_dets(bgr, preds)
                img_ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                json_ph.json(data)
                st.success(f"Detections: {len(preds)}")
            except Exception as e:
                st.error(f"Error: {e}")

    elif submode == "Upload Image":
        up = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png", "bmp"])
        if up and st.button("Run Detection"):
            pil = Image.open(up).convert("RGB")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=90)
            try:
                data = infer_bytes(buf.getvalue())
                preds = data.get("predictions", [])
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                vis = draw_dets(bgr, preds)
                img_ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                json_ph.json(data)
                st.success(f"Detections: {len(preds)}")
            except Exception as e:
                st.error(f"Error: {e}")

    else:  
        url_inp = st.text_input("رابط صورة مباشر (jpg/png)")
        if url_inp and st.button("Run Detection"):
            try:
                r = requests.get(url_inp, timeout=15)
                r.raise_for_status()
                pil = Image.open(io.BytesIO(r.content)).convert("RGB")
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=90)
                data = infer_bytes(buf.getvalue())
                preds = data.get("predictions", [])
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                vis = draw_dets(bgr, preds)
                img_ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                json_ph.json(data)
                st.success(f"Detections: {len(preds)}")
            except Exception as e:
                st.error(f"Error: {e}")
