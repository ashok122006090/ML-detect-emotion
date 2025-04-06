from fastapi import FastAPI, Request
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_methods=["*"],
   allow_headers=["*"],
)

# Load the Hugging Face emotion model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

@app.get("/")
def root():
    return {"message": "Emotion detection API is live!"}

@app.post("/detect-emotion")
async def detect_emotion(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    result = classifier(text)
    return {"emotion": result[0]["label"]}
