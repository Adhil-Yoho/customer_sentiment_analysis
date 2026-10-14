import torch

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_PATH = "outputs/distilbert_sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

app = FastAPI(title="Customer Feedback Classifier")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)



@app.post("/predict")
def predict(request: PredictionRequest):

    text = request.text.strip()

    if not text:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Text must not be empty or whitespace."
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(
            outputs.logits, dim=1
        )[0]

    predicted_id = int(torch.argmax(probabilities).item())

    label = model.config.id2label.get(
        predicted_id, str(predicted_id)
    )

    confidence = float(probabilities[predicted_id].item())

    return {
        "text": text,
        "label": label,
        "confidence": round(confidence, 4)
    }
