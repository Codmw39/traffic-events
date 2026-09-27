"""Minimal live-demo app for the website's 'Live demo' requirement.
Deploy to Hugging Face Spaces (free CPU tier is fine per the task rules) and
link to it from the main website, or embed it in an <iframe>.

To deploy on HF Spaces:
  1. Create a new Space, SDK = Streamlit.
  2. Upload this file as app.py, plus solution.py, src/, weights/, zones.json,
     and a requirements.txt that also includes: streamlit
  3. The Space builds and gives you a public URL - link/embed that.
"""
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

import solution

st.set_page_config(page_title="Traffic Event Detection - Live Demo", layout="centered")
st.title("Traffic event detection — live demo")
st.caption("Upload a short clip (\u2264 2 minutes) from the same fixed camera angle.")

uploaded = st.file_uploader("Upload .mp4", type=["mp4"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if duration > 130:
        st.error("Please upload a clip of 2 minutes or less for the demo.")
    else:
        st.video(video_path)
        progress = st.progress(0, text="Detecting events (Part A)...")
        t0 = time.time()
        events = solution.detect_events(video_path)
        progress.progress(50, text="Estimating risk (Part B)...")

        est = solution.RiskEstimator()
        est.reset({"video_id": uploaded.name, "fps": fps, "width": width,
                   "height": height, "n_frames": n_frames})
        risk_curve = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = idx / fps
            score = est.step(frame, t_sec)
            risk_curve.append(score)
            idx += 1
            if idx % 30 == 0:
                progress.progress(50 + int(50 * idx / max(n_frames, 1)),
                                  text=f"Estimating risk (Part B)... {idx}/{n_frames}")
        cap.release()
        progress.progress(100, text="Done")
        st.success(f"Processed in {time.time() - t0:.1f}s")

        st.subheader("Detected events")
        if events:
            st.table([{"start (s)": round(s, 1), "end (s)": round(e, 1), "event": label}
                      for s, e, label in sorted(events)])
        else:
            st.info("No events detected in this clip.")

        st.subheader("Accident risk over time")
        st.line_chart(risk_curve)

        Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video to see detected events and the risk curve.")
