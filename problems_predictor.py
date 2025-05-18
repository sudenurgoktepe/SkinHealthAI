import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pickle

# Cilt Problemi Modeli
def load_problem_model():
    model_path = "./bert-problem-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    with open("le_problems.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# Cilt Tipi Modeli
def load_ciltbert_model():
    model_path = "./bert-cilt-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    with open("le_cilt2.pkl", "rb") as f:  # <-- DÜZENLENDİ
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# Cilt Problemi Tahmin
def predict_problem(text, model, tokenizer, label_encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()


    if pred_class >= len(label_encoder.classes_):
        label = "Bilinmeyen"
        confidence = 0.0
    else:
        label = label_encoder.inverse_transform([pred_class])[0]

    return label, confidence

# Cilt Tipi Tahmin
def predict_ciltbert(text, model, tokenizer, label_encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()


    if pred_class >= len(label_encoder.classes_):
        label = "Bilinmeyen"
        confidence = 0.0
    else:
        label = label_encoder.inverse_transform([pred_class])[0]

    return label, confidence
