
import re
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

positive_messages = [
    "I love this product",
    "The service was excellent",
    "Very happy with my purchase",
    "The delivery was fast",
    "Great quality and good value",
    "The staff were friendly",
    "Everything worked perfectly",
    "I would recommend this product",
    "The app is easy to use",
    "Excellent customer support"
]

negative_messages = [
    "I hate this product",
    "The service was terrible",
    "Very unhappy with my purchase",
    "The delivery was late",
    "Poor quality and bad value",
    "The staff were rude",
    "Nothing worked properly",
    "I would not recommend this product",
    "The app is difficult to use",
    "Terrible customer support"
]

positive_starts = ["", "Today ", "Honestly ", "Overall ", "Really "]
negative_starts = ["", "Today ", "Honestly ", "Overall ", "Really "]
positive_ends = ["", " Highly recommended.", " I am satisfied.", " Thank you."]
negative_ends = ["", " Not recommended.", " I am disappointed.", " Very frustrating."]

texts = []
labels = []

for i in range(20):
    for message in positive_messages:
        text = (
            positive_starts[i % len(positive_starts)]
            + message
            + positive_ends[i % len(positive_ends)]
        )
        texts.append(text)
        labels.append("positive")

    for message in negative_messages:
        text = (
            negative_starts[i % len(negative_starts)]
            + message
            + negative_ends[i % len(negative_ends)]
        )
        texts.append(text)
        labels.append("negative")

df = pd.DataFrame({"text": texts, "label": labels})
df.to_csv("feedback_dataset.csv", index=False)

print("Dataset shape:", df.shape)
print(df.head())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

start_time = time.time()
predictions = model.predict(X_test)
inference_time = (time.time() - start_time) / len(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

print("\nTF-IDF results")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Training time: {training_time:.4f} seconds")
print(f"Average inference time: {inference_time:.6f} seconds")
print("Labels:", model.named_steps["classifier"].classes_)
print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=model.named_steps["classifier"].classes_
))


vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

words = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]
classes = classifier.classes_


print("\nImportant words for", classes[1])
for index in np.argsort(coefficients)[-10:][::-1]:
    print(words[index], round(coefficients[index], 3))

print("\nImportant words for", classes[0])
for index in np.argsort(coefficients)[:10]:
    print(words[index], round(coefficients[index], 3))


# --------------------------------------------------
# 5. Load pretrained GloVe embeddings
# --------------------------------------------------

try:
    import gensim.downloader as api

    print("\nLoading GloVe vectors...")
    glove = api.load("glove-wiki-gigaword-50")

    for word in ["king", "computer"]:
        print(f"\nWords similar to {word}:")
        print(glove.most_similar(word, topn=5))

    analogy = glove.most_similar(
        positive=["king", "woman"],
        negative=["man"],
        topn=1
    )
    print("\nking - man + woman:", analogy[0][0])
    
    def sentence_vector(text):
            vectors = [
                glove[word]
                for word in text.split()
                if word in glove
            ]
    
            if not vectors:
                return np.zeros(50)
    
            return np.mean(vectors, axis=0)
    
    X_train_vectors = np.vstack([
            sentence_vector(text) for text in X_train
        ])
    
    X_test_vectors = np.vstack([
            sentence_vector(text) for text in X_test
        ])
    
    embedding_model = LogisticRegression(max_iter=1000)
    
    start_time = time.time()
    embedding_model.fit(X_train_vectors, y_train)
    embedding_training_time = time.time() - start_time
    
    embedding_predictions = embedding_model.predict(X_test_vectors)
    
    embedding_accuracy = accuracy_score(
            y_test, embedding_predictions
        )
    embedding_f1 = f1_score(
            y_test, embedding_predictions, average="macro"
        )
    
    print("\nAveraged GloVe sentence vectors")
    print(f"Accuracy: {embedding_accuracy:.4f}")
    print(f"Macro F1: {embedding_f1:.4f}")
    print(f"Training time: {embedding_training_time:.4f} seconds")
    
except Exception as error:
    print("\nGloVe step could not run:", error)
    print("Check your internet connection and Gensim installation.")