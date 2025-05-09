# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
import io
import os

app = FastAPI()

# Allow mobile app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(
    model_name,
    token=HF_TOKEN
).to(device)

tokenizer = CLIPTokenizer.from_pretrained(model_name)
image_processor = CLIPImageProcessor.from_pretrained(model_name, use_fast=True)

# Categories
labels = [
  "an image of pornography", 
  "an image of nudity", 
  "sexual content", 
  "graphic violence", 
  "a normal safe photo", 
  "self-harming behavior", 
  "a completely safe image"
]

# Prepare text inputs (labels)
text_inputs = tokenizer(labels, return_tensors="pt", padding=True).to(device)

@app.get("/")
def root():
    return {"message": "ClearPath AI server running."}

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prepare image input
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        
        # Run the model
        outputs = model(**inputs, **text_inputs)
        
        # Calculate probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
        print("Raw model outputs (probabilities):", probs.tolist())

        # Calculate percentage results
        percent_results = {
            label: round(float(prob) * 100, 2) for label, prob in zip(labels, probs)
        }

        # Get top match
        top_label = max(percent_results, key=percent_results.get)
        top_confidence = percent_results[top_label]

        # Define threshold for flagging
        risky_labels = ["porn", "nude", "sexual", "violence", "self-harm"]
        is_flagged = top_label in risky_labels and top_confidence >= 25

        # Return the results
        return {
            "top_prediction": top_label,
            "confidence_percent": f"{top_confidence}%",
            "flagged_as_harmful": is_flagged,
            "full_results_percent": percent_results
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e)}