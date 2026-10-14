# Deep Learning + NLP: Week 3 Assessment

This project covers neural networks, PyTorch training pipelines, classical NLP, transformer fine-tuning, and model serving with FastAPI. It has four tasks:

1. Neural network fundamentals with NumPy.
2. A PyTorch training pipeline on the Breast Cancer dataset.
3. Classical NLP with TF-IDF, Logistic Regression, and GloVe embeddings.
4. Transformers with Hugging Face DistilBERT and FastAPI.

## Technologies used

- Python
- NumPy
- Matplotlib
- PyTorch
- Pandas
- Scikit-learn
- Gensim and GloVe
- Hugging Face Transformers and Datasets
- FastAPI and Uvicorn

## Project structure

The project uses Python scripts and an `outputs` folder for generated results and screenshots.

```
customer_sentiment_analysis/
│
├── neural_network.py
├── pytorch.py
├── nlp.py
├── train_transformer.py
├── api.py
├── feedback_dataset.csv
├── loss_curve.png
├── learning_rate_comparison.png
├── cancer_mlp.pth
├── outputs/
│ ├── model_comparison.csv
│ └── API response screenshot
└── README.md
```

The filenames above are examples. Keep your actual Python filenames and folder structure. The generated dataset and saved model files will appear wherever your scripts save them.

## Task 1: Neural network fundamentals

**Objective.** Build and train a 2-4-1 neural network in NumPy to solve the XOR classification problem.

### Implementation

- Implemented ReLU, Sigmoid, Softmax, Mean Squared Error (MSE), and Binary Cross-Entropy functions.
- Built a network with two input neurons, four hidden neurons, and one output neuron.
- Trained it for 5,000 epochs using gradient descent with a learning rate of 0.5.
- Wrote forward propagation and backpropagation by hand.
- Plotted the training loss and compared different learning rates.
- Calculated gradients manually and checked them against PyTorch autograd.

### Results

Softmax:

```
[0.659 0.242 0.099]
```

XOR predictions:

```
[0. 1. 1. 0.]
```

Predicted classes:

```
[0 1 1 0]
```

Gradient verification:

```
Manual gradients:
dw = -32
db = -16

PyTorch autograd:
w.grad = -32.0
b.grad = -16.0
```

The network classified all four XOR examples correctly.

### Regularisation concepts

- **Dropout** randomly switches off a share of neurons during training to reduce overfitting. It is turned off during evaluation.
- **BatchNorm** normalizes layer activations using batch statistics during training, which can steady and speed up training.
- **Early stopping** watches validation performance and stops training once it stops improving. This reduces overfitting and wasted training time.

## Task 2: PyTorch training pipeline

**Objective.** Build, train, evaluate, save, and reload a PyTorch neural network, then compare it with Logistic Regression.

### Implementation

- Used the Scikit-learn Breast Cancer dataset, which has 569 samples and 30 features.
- Split the data into training, validation, and test sets.
- Applied `StandardScaler`, fitted only on the training data.
- Wrote a custom `CsvDataset` class on top of PyTorch's `Dataset` interface.
- Loaded data with `DataLoader` at a batch size of 32.
- Built an MLP from Linear layers, ReLU, Dropout, and BatchNorm1d.
- Trained with Adam and `BCEWithLogitsLoss`.
- Tracked training loss and validation accuracy, with early stopping.
- Saved the model's `state_dict` and reloaded it into a new model.
- Compared the MLP with Scikit-learn Logistic Regression.

### Results

| Metric | Result |
|---|---|
| Training samples | 341 |
| Validation samples | 114 |
| Test samples | 114 |
| Model parameters | 1,633 |
| Best validation accuracy | 98.25% |
| MLP test accuracy | 95.61% |
| Logistic Regression test accuracy | 99.12% |
| Reloaded model accuracy | 95.61% |
| Accuracy identical after reloading | True |

The reloaded model matched the original model's test accuracy. On this split, Logistic Regression scored higher than the MLP.

## Task 3: Classical NLP baseline and word embeddings

**Objective.** Build a sentiment classifier with TF-IDF and Logistic Regression, then compare it with averaged GloVe sentence embeddings.

### Implementation

- Created a synthetic dataset of 400 labelled customer-feedback messages.
- Cleaned the text by lowercasing it and removing punctuation.
- Split the data into training and test sets with stratified sampling.
- Built a TF-IDF and Logistic Regression pipeline.
- Evaluated it with accuracy, macro F1, and a confusion matrix.
- Printed the ten most important words for each sentiment class.
- Loaded pretrained 50-dimensional GloVe word embeddings.
- Found the five nearest words to "king" and "computer".
- Solved the analogy "king - man + woman".
- Averaged word vectors into sentence representations and compared their classification performance with TF-IDF.

### Results

| Metric | TF-IDF + Logistic Regression | Averaged GloVe vectors |
|---|---|---|
| Accuracy | 97.50% | 95.00% |
| Macro F1 | 0.9750 | 0.9500 |

Confusion matrix for TF-IDF:

```
[[39 1]
 [ 1 39]]
```

Word embedding analogy:

```
king - man + woman -> queen
```

TF-IDF beat averaged GloVe vectors on this dataset. The data is synthetic and built from repeated templates, so these results don't reflect how the models would do on real customer feedback.

## Task 4: Transformers, fine-tuning, and serving

**Objective.** Explore transformer tokenization and pretrained pipelines, fine-tune DistilBERT for sentiment classification, compare it with a classical baseline, and serve predictions through FastAPI.

### Implementation

- Loaded the `distilbert-base-uncased` tokenizer.
- Displayed tokens, input IDs, and attention masks for two sentences.
- Ran the Hugging Face sentiment-analysis pipeline on three sentences.
- Ran the named entity recognition (NER) pipeline on three sentences.
- Fine-tuned DistilBERT for two epochs with the Hugging Face Trainer API.
- Evaluated it with accuracy and macro F1.
- Compared training time and inference time against TF-IDF + Logistic Regression.
- Saved the fine-tuned model and tokenizer.
- Built a FastAPI `POST /predict` endpoint that returns a predicted label and confidence.
- Added validation that returns HTTP 422 for empty text.

### Transformer results

| Metric | TF-IDF + Logistic Regression | Fine-tuned DistilBERT |
|---|---|---|
| Accuracy | 97.50% | 100.00% |
| Macro F1 | 0.9750 | 1.0000 |
| Training time | 0.1027 seconds | 30.2338 seconds |
| Inference time per text | About 0.000025 seconds | 0.0081 seconds |

These numbers come from the current synthetic dataset and execution environment. TF-IDF trained much faster, while DistilBERT scored higher. The data is built from repeated templates, and the test set was also used for evaluation during training, so these scores don't show how the models would generalize to unseen real-world feedback.

### FastAPI endpoint

`POST /predict`

Example request:

```json
{
  "text": "The product is excellent and the service was great."
}
```

Example response:

```json
{
  "text": "The product is excellent and the service was great.",
  "label": "positive",
  "confidence": 0.6717
}
```

The response above shows the expected format. The actual confidence depends on the trained model. Empty or whitespace-only input is rejected with HTTP 422.

To start the API, run this from the directory containing `api.py` and the saved model folder:

```bash
uvicorn api:app --reload
```

Then open the interactive documentation at http://127.0.0.1:8000/docs.

The `outputs` folder holds the model-comparison results and the API response screenshot for submission.

## BERT (encoder) vs GPT (decoder)

BERT is an encoder-based transformer. It learns contextual representations from the tokens on both sides of each word, which makes BERT-style models a good fit for classification, sentiment analysis, and named entity recognition.

GPT is a decoder-based transformer. It predicts the next token, so GPT-style models are used for text generation, conversational responses, and other generative tasks.

For customer-feedback classification, I would start with TF-IDF and Logistic Regression as a baseline. I would move to DistilBERT only if its measured performance justifies the extra computing cost.

## Key concepts learned

- Neural network forward propagation and backpropagation.
- Activation functions, loss functions, and gradient descent.
- Learning rates and the effect of regularisation.
- PyTorch tensors, `nn.Module`, Dataset, DataLoader, and training loops.
- Model evaluation, early stopping, and saving and loading a `state_dict`.
- Text preprocessing, TF-IDF, Logistic Regression, and confusion matrices.
- Word embeddings, GloVe similarity, and sentence-vector averaging.
- Hugging Face tokenizers, pipelines, transformer fine-tuning, and evaluation.
- FastAPI endpoints, request validation, and model inference.

## Conclusion

This project walks through an introductory deep learning and NLP workflow. It starts with a neural network built from scratch, then moves through PyTorch models, a comparison of text-classification methods, a fine-tuned transformer, and a REST API that serves predictions.

The experiments also show why a simple TF-IDF baseline is worth building. It gave competitive results at a much lower training and inference cost. Choosing a model should depend on measured performance, how well it generalizes, and what it costs to run.
