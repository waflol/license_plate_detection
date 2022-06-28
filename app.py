

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from tools.image import *
import tempfile

device = "cpu"
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/plate_number_detection.pt')
model = model.to(device)
model = model.eval()

st.title("License plate detection and recognition")
instructions = """
    Người thực hiện:
    - Lê Nguyễn Hùng Anh - 19110322
    - Phan Tấn Thành - 19110288
    - Phạm Ngọc Đức - 19110157
    """
st.write(instructions)

file = st.file_uploader('Upload An Image',type = ['jpg','png','jpeg'])

if file:
    img = Image.open(file)
    img = np.array(img)
    img = cv2.resize(img,(512,512),cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
    results = model(img)
    if results.xyxy[0].size()[0] != 0:
        st.caption('Detection license plates')
        xmin, ymin, xmax, ymax = results.xyxy[0][0].cpu().numpy()[:4]
            
        draw_bbox_img = draw_bbox(img.copy(),results.xyxy[0][0].cpu().numpy()[:4])
        st.caption('Results detection')
        st.image(draw_bbox_img,channels='BGR')
        
        ROI_results = extractROIimg_fromPoints(img.copy(),results.xyxy[0][0].cpu().numpy()[:4])
        imgGrayscale, imgThresh = preprocess(ROI_results.copy())
        col1_1, col1_2,col1_3 = st.columns(3)
        with col1_1:
            st.caption('Gray scale')
            st.image(imgGrayscale)
        with col1_2:
            st.caption('Thresh')
            st.image(imgThresh)
        with col1_3:
            plate_number_recognize = detect_character(ROI_results.copy(),imgThresh.copy())
            st.caption('Results recognization')
            st.image(plate_number_recognize,channels='BGR')
    else:
        st.caption('No detection!')
    

