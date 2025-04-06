from fastapi import FastAPI, Request
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load the scikit-learn model
model = joblib.load("emotion_model.pkl")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Emotion detection API is up!"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")

    if not text:
        return {"error": "No text provided"}

    prediction = model.predict([text])[0]
    return {"emotion": prediction}
