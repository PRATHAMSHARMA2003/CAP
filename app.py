import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# Load the trained model
model = YOLO('best.pt')  # Replace with the actual path to your trained model

# Streamlit app title
st.title('Food Detection using YOLOv8')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Perform inference
    image_array = np.array(image)
    results = model(image_array)
    
    # Print the results
    st.write("Detected objects:")
    st.write(results.pandas().xyxy[0])  # DataFrame of detection results
    
    # Visualize the results
    results_img = results.render()[0]  # Render the detections
    st.image(results_img, caption='Detected Image', use_column_width=True)
