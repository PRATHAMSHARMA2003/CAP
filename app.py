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
    # Open and preprocess the image
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # Convert to RGB mode
    image = np.asarray(image)  # Convert PIL image to numpy array
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Perform inference
    results = model(image)  # Pass the preprocessed image array
    
    # Get the detection results
    detections = results.xyxy[0] if len(results.xyxy) > 0 else []
    
    # Check if any detections were made
    if len(detections) > 0:
        # Display detected objects
        st.write("Detected objects:")
        for detection in detections:
            class_name = model.names[int(detection[-1])]
            confidence = detection[-2]
            st.write(f"Class: {class_name}, Confidence: {confidence:.2f}")
    else:
        st.write("No objects detected.")
