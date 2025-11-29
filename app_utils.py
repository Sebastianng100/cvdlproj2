# app_utils.py
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import cv2


MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    model.model.names = {
        0: "Fall",
        1: "Stand"
    }
    return model


def run_inference_on_image(model, image: Image.Image, conf: float = 0.5):
    img_rgb = np.array(image)

    results = model.predict(img_rgb, conf=conf, verbose=False)
    result = results[0]

    # Render YOLO annotations
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        cls_name = model.model.names.get(cls_id, str(cls_id))
        detections.append((cls_name, conf_score))

    return annotated_pil, detections


def save_uploaded_file_to_temp(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def process_video_file(model, input_path, output_path, conf=0.5, frame_stride=1):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            results = model.predict(frame, conf=conf, verbose=False)
            annotated = results[0].plot()
        else:
            annotated = frame

        out.write(annotated)
        frame_idx += 1
        progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    out.release()
    progress.empty()
