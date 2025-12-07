## Prototype Development for Image Captioning Using the BLIP Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

### PROBLEM STATEMENT:
he goal of this project is to build a simple prototype that can automatically generate captions for images. Although powerful models like BLIP exist for image captioning, using them directly through APIs can be difficult for beginners. To make this process easier, this experiment integrates the BLIP model with a Gradio interface so users can upload an image and instantly receive a caption. This project demonstrates how to process an image, send it to the model, and display the generated caption in an easy-to-use web app.

### DESIGN STEPS:

#### **1️⃣ Load Environment & Dependencies**
Install required libraries and verify GPU support (CUDA).  
This ensures the model runs efficiently on Google Colab without external APIs.

#### **2️⃣ Load BLIP Model Using Hugging Face**
Use the `pipeline("image-to-text")` with `Salesforce/blip-image-captioning-base`.  
The pipeline automatically loads tokenizer, processor, and model.

#### **3️⃣ Create Caption Generation Function**
Accept an uploaded image → process using the BLIP pipeline → return the model’s generated caption.

#### **4️⃣ Build Simple UI with Gradio**
Provide an image upload field and text output box.  
Add optional sample images for quick testing.

#### **5️⃣ Run and Test the Application**
Launch Gradio interface, upload images, and validate captions on Colab GPU.  
Evaluate accuracy, speed, and model response quality.


### PROGRAM:

```py
import torch
from transformers import pipeline
import gradio as gr
import io
import base64
from PIL import Image

# Load model on GPU if available
device = 0 if torch.cuda.is_available() else -1
print("Running on:", "GPU" if device == 0 else "CPU")

pipe = pipeline("image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=device)

# Gradio function
def captioner(image):
    if image is None:
        return "Please upload an image!"
    result = pipe(image)
    return result[0]['generated_text']

# Create UI
gr.close_all()
demo = gr.Interface(
    fn=captioner,
    inputs=gr.Image(label="Upload image", type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="BLIP Image Captioning (Local GPU)",
    description="Generates image captions locally using BLIP model.",
    allow_flagging="never",
    examples=["/content/16008_PetPhotos_MC.jpg", "/content/f2672b840d9a0c45306167e6a56261fc.jpg"]
)

demo.launch()
```

### OUTPUT:

<img width="1918" height="1154" alt="image" src="https://github.com/user-attachments/assets/84ea8e0a-1011-4a63-822d-0243d57dcced" />

### RESULT:

Thus, the image captioning application using the BLIP model was successfully developed and deployed with Gradio. The system efficiently generates descriptive captions for images, demonstrating the practical use of multimodal AI in automation and accessibility.
