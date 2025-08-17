import streamlit as st
from ultralytics import YOLO
import cv2
import math
import cvzone
import tempfile
import numpy as np
import base64
import smtplib
from playsound import playsound

# Email setup
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login("smamatha1509@gmail.com", "nergiekbwpyoobpj")
message = "HUMAN IS DETECTED"

def home_page():
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title('Welcome to the Drone Detection System')
    st.markdown("""
        <div class='big-font'>This application uses advanced YOLOV8 techniques to detect Drone in images, videos, and live webcams.</div>
    """, unsafe_allow_html=True)

def set_background_image(image_file):
    with open(image_file, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image('basic3.jpg')

def main_app():
    model = YOLO('yolov8n.pt')
    classnames = model.names
    st.title('Drone Human Detection')

    input_option = st.radio("Choose input type", ('Image', 'Video', 'Live Stream'))

    def process_frame(frame):
        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if Class >= len(classnames):
                    continue
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                    if classnames[Class] == "person":
                        st.write("Human detected")
                        s.sendmail("smamatha1509@gmail.com", "smamatha1309@gmail.com", message)
                        playsound('3.wav')
                    else:
                        st.write("Not a Human detected")
        return frame

    if input_option == 'Image':
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            processed_image = process_frame(image)
            st.image(processed_image, channels="BGR", use_column_width=True)
    
    elif input_option == 'Video':
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            frameST = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(frame)
                frameST.image(processed_frame, channels="BGR", use_column_width=True)
            cap.release()

    elif input_option == 'Live Stream':
        st.write("Live Stream from Webcam")
        frame_processed = st.image([])
        cap = cv2.VideoCapture(0)
        
        stop_streaming = st.button('Stop Streaming', key="stop_live_stream")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_streaming:
                break
            processed_frame = process_frame(frame)
            frame_processed.image(processed_frame, channels='BGR')
        
        cap.release()

home_page()
main_app()
