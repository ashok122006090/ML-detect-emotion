from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import warnings

# Suppress known warnings during deployment
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="resume_download is deprecated")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend/backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face emotion classification model
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

@app.get("/")
def root():
    return {"message": "Emotion Detection API is live!"}

@app.post("/detect-emotion")
async def detect_emotion(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    
    result = classifier(text)
    return {"emotion": result[0]["label"]}
