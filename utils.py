#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import torch


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    import torch
    from ultralytics.nn.tasks import torch_safe_load
      
      # Patch the torch_safe_load function
    def patched_torch_safe_load(weight):
       return torch.load(weight, map_location='cpu', weights_only=False), weight
      
      # Replace the original function with the patched version
    torch_safe_load = patched_torch_safe_load
    model = YOLO(model_path)
    return model

def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

def infer_rtsp_stream(conf, model):
    """
    Execute inference for RTSP stream with enhanced error handling
    :param conf: Confidence threshold
    :param model: YOLOv8 model instance
    """
    st.subheader("RTSP Stream Configuration")
    
    # URL input with validation
    rtsp_url = st.text_input(
        "RTSP Stream URL",
        placeholder="rtsp://username:password@ip:port/path",
        help="Example: rtsp://admin:123456@192.168.1.100:554/stream"
    )
    
    # Authentication options
    with st.expander("Advanced Settings"):
        use_tcp = st.checkbox("Force TCP Protocol", True)
        buffer_size = st.slider("Stream Buffer Size", 1, 10, 3)
    
    # Stream control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start Stream", type="primary")
    with col2:
        stop_btn = st.button("Stop Stream", type="secondary")

    # Initialize session state
    if 'stream_active' not in st.session_state:
        st.session_state.stream_active = False
        st.session_state.vid_cap = None

    # Handle button actions
    if start_btn and rtsp_url:
        st.session_state.stream_active = True
    if stop_btn:
        st.session_state.stream_active = False
        if st.session_state.vid_cap:
            st.session_state.vid_cap.release()

    # Main stream processing
    if st.session_state.stream_active:
        try:
            # Build connection parameters
            connection_params = {
                'rtsp_transport': 'tcp' if use_tcp else 'udp',
                'buffer_size': 1024 * 1024 * buffer_size
            }
            
            # Initialize video capture
            if not st.session_state.vid_cap:
                st.session_state.vid_cap = cv2.VideoCapture(rtsp_url)
                st.session_state.vid_cap.set(
                    cv2.CAP_PROP_BUFFERSIZE, 
                    connection_params['buffer_size']
                )

            # Check connection status
            if not st.session_state.vid_cap.isOpened():
                st.error("Failed to connect to RTSP stream!")
                st.session_state.stream_active = False
                return

            # Create stream container
            stream_placeholder = st.empty()
            status_text = st.empty()
            warning_placeholder = st.empty()
            
            # Frame processing loop
            while st.session_state.stream_active:
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    success, frame = st.session_state.vid_cap.read()
                    
                    if not success:
                        retry_count += 1
                        warning_placeholder.warning(
                            f"Frame read error, retrying ({retry_count}/{max_retries})..."
                        )
                        # Reset connection on failure
                        st.session_state.vid_cap.release()
                        st.session_state.vid_cap = cv2.VideoCapture(rtsp_url)
                        continue
                        
                    # Successful frame read
                    retry_count = 0
                    warning_placeholder.empty()
                    
                    # Perform detection
                    res = model.predict(
                        source=frame, 
                        conf=conf, 
                        verbose=False
                    )
                    res_plotted = res[0].plot()[:, :, ::-1]
                    
                    # Display processed frame
                    stream_placeholder.image(
                        res_plotted,
                        caption="Live RTSP Stream",
                        channels="BGR",
                        use_column_width=True
                    )
                    
                    # Update status
                    status_text.info(f"Stream Status: Active | Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    break
                
                else:
                    st.error("Maximum retry attempts reached. Stopping stream.")
                    st.session_state.stream_active = False

        except Exception as e:
            st.error(f"Stream Error: {str(e)}")
            st.session_state.stream_active = False

        finally:
            if st.session_state.vid_cap:
                st.session_state.vid_cap.release()
                st.session_state.vid_cap = None
            st.session_state.stream_active = False
            status_text.success("Stream successfully stopped")
