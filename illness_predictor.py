# illness_predictor.py

import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# HASTALIK MODELİ
def load_illness_model():
    model = load_model('hastalik_model_best_bilstm.keras')  # ✅ Güncel ve doğru
    with open('tokenizer_hastalik.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('le_hastalik.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

def predict_illness(text, model, tokenizer, label_encoder):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    preds = model.predict(padded)
    pred_class = np.argmax(preds)
    confidence = np.max(preds)
    class_name = label_encoder.inverse_transform([pred_class])[0]
    return class_name, confidence

# CİLT TİPİ MODELİ
def load_cilt_model():
    model = load_model('cilttipi_model_best_bilstm.keras')  # ✅ Güncel ve doğru
    with open('tokenizer_cilt.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('le_cilt.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

def predict_cilt(text, model, tokenizer, label_encoder):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    preds = model.predict(padded)
    pred_class = np.argmax(preds)
    confidence = np.max(preds)
    class_name = label_encoder.inverse_transform([pred_class])[0]
    return class_name, confidence
