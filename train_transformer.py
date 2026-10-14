
import os
import time
import numpy as np
import pandas as pd
import torch

from transformers import (AutoTokenizer, AutoModelForSequenceClassification, pipeline,Trainer, TrainingArguments, DataCollatorWithPadding)

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

sentences = [
    "The service was excellent.",
    "I am unhappy with the delivery."
]

for sentence in sentences:
    result = tokenizer(sentence)

    print("\nSentence:", sentence)
    print("Tokens:", tokenizer.convert_ids_to_tokens(result["input_ids"]))
    print("Input IDs:", result["input_ids"])
    print("Attention mask:", result["attention_mask"])


sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment_sentences = [
    "I love the quality of this product.",
    "The customer service was terrible.",
    "The product arrived today."
]

print("\nSentiment analysis results:")
for sentence in sentiment_sentences:
    print(sentence)
    print(sentiment_pipeline(sentence))


ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

ner_sentences = [
    "Microsoft is based in the United States.",
    "Sundar Pichai works at Google.",
    "Paris is the capital of France."
]

print("\nNER results:")
for sentence in ner_sentences:
    print(sentence)
    print(ner_pipeline(sentence))


df = pd.read_csv("feedback_dataset.csv")

df["text"] = df["text"].fillna("").astype(str)
df["label_id"] = df["label"].map({
    "negative": 0,
    "positive": 1
})

df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

train_df, test_df = train_test_split(
    df[["text", "label_id"]],
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

train_dataset = Dataset.from_pandas(
    train_df.reset_index(drop=True)
)

test_dataset = Dataset.from_pandas(
    test_df.reset_index(drop=True)
)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(
    tokenize_function, batched=True
)

test_dataset = test_dataset.map(
    tokenize_function, batched=True
)

train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

columns = ["input_ids", "attention_mask", "labels"]

train_dataset.set_format(type="torch", columns=columns)
test_dataset.set_format(type="torch", columns=columns)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1}
)


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels, predictions, average="macro"
        )
    }


output_dir = "./outputs/distilbert_sentiment"

training_args = TrainingArguments(
    output_dir="./outputs/training_checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

start_time = time.time()
trainer.train()
transformer_training_time = time.time() - start_time


results = trainer.evaluate()
transformer_accuracy = results["eval_accuracy"]
transformer_f1 = results["eval_f1"]

print("\nDistilBERT results:")
print(f"Accuracy: {transformer_accuracy:.4f}")
print(f"Macro F1: {transformer_f1:.4f}")
print(f"Training time: {transformer_training_time:.2f} seconds")

test_texts = test_df["text"].tolist()

start_time = time.time()

predictions_output = trainer.predict(test_dataset)
predicted_labels = np.argmax(
    predictions_output.predictions, axis=1
)

transformer_inference_time = (
    time.time() - start_time
) / len(test_texts)


baseline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

start_time = time.time()
baseline.fit(train_df["text"], train_df["label_id"])
baseline_training_time = time.time() - start_time

start_time = time.time()
baseline_predictions = baseline.predict(test_df["text"])
baseline_inference_time = (
    time.time() - start_time
) / len(test_texts)

baseline_accuracy = accuracy_score(
    test_df["label_id"], baseline_predictions
)

baseline_f1 = f1_score(
    test_df["label_id"],
    baseline_predictions,
    average="macro"
)



comparison = pd.DataFrame([
    {
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy": baseline_accuracy,
        "Macro F1": baseline_f1,
        "Training time (seconds)": baseline_training_time,
        "Inference time per text (seconds)": baseline_inference_time
    },
    {
        "Model": "Fine-tuned DistilBERT",
        "Accuracy": transformer_accuracy,
        "Macro F1": transformer_f1,
        "Training time (seconds)": transformer_training_time,
        "Inference time per text (seconds)": transformer_inference_time
    }
])

print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))

os.makedirs("./outputs", exist_ok=True)
comparison.to_csv("./outputs/model_comparison.csv", index=False)


os.makedirs(output_dir, exist_ok=True)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("\nSaved model to:", output_dir)
print("Saved comparison to: ./outputs/model_comparison.csv")

