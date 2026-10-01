from ultralytics import YOLO
import gradio as gr
import numpy as np

model = YOLO("best.pt")

def predict_image(image):
    if image is None:
        return None, None
    results = model.predict(image)
    plotted = results[0].plot()
    return plotted, None  # return image result, no video

def predict_video(video):
    if video is None:
        return None, None
    results = model.predict(video)
    # For video, return the last frame's plot as image preview
    plotted = results[0].plot()
    return None, video  # return original video (annotated video needs more work)

def predict(image, video):
    if image is not None:
        return predict_image(image)
    elif video is not None:
        return predict_video(video)
    return None, None

app = gr.Interface(
    fn=predict,
    inputs=["image", "video"],
    outputs=["image", "video"],
    title="Number Plate Detection",
    description="Upload an image or video to detect number plates."
)

app.launch(share=True)
