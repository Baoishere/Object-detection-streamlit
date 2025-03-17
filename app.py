#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Description:
-------------------------------------------------
"""
from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam, infer_rtsp_stream

import os

# model_path = "weights/detection/yolov8n.pt"  # Đường dẫn model
# if os.path.exists(model_path):
#     file_size = os.path.getsize(model_path) / (1024*1024)  # Đổi sang MB
#     print(f"📏 Model Size: {file_size:.2f} MB")
# else:
#     print("❌ Model file not found!")


# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Interactive Interface for YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

try:
    model_path = "weights/detection/yolov8n.pt"

    # Load model với weights_only=False
    model = torch.load(model_path, weights_only=False)  # ⚠️ Chỉ làm nếu file từ nguồn đáng tin cậy

    # Hoặc sử dụng YOLO của Ultralytics
    model = YOLO(model_path)  
    model.to("cpu")  # Chạy trên CPU
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")


# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
elif source_selectbox == config.SOURCES_LIST[3]: # RTSP Stream
    infer_rtsp_stream(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
