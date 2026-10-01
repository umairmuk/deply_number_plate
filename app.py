from ultralytics import YOLO
import streamlit as st

# YOLO model load
model = YOLO("best.pt")

st.title("YOLO Object Detection")

# Image upload
image = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if image is not None:

    # YOLO prediction
    results = model.predict(image)

    # Detection result plot
    result_image = results[0].plot()

    # Show result
    st.image(result_image, caption="Detection Result")
