import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the trained model
model = YOLO('best.pt')  # Update the path to your saved model

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
    
    # Check if any detections were made
    if results.xyxy:
        # Process detection results
        detections = results.xyxy[0]
        
        # Display detected objects
        st.write("Detected objects:")
        for detection in detections:
            class_name = model.names[int(detection[5])]
            confidence = detection[4]
            st.write(f"Class: {class_name}, Confidence: {confidence:.2f}")
    else:
        st.write("No objects detected.")
