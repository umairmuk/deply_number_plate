from ultralytics import YOLO
import gradio as gr 


model = YOLO("best.pt")

def pred_image(image):
    img = model.predict(image)
    return img[0].plot()


app= gr.Interface(fn = pred_image, inputs = ["image", 'video'], outputs = ["image", 'video'])
app.launch(share=True)


# python  app.py
# py app.py

#ctrl c   #   reset the kernal 
