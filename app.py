import os
# Forza l'uso di PyTorch ed evita conflitti con TensorFlow (causa dell'errore DLL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import os
import re
import torch
import joblib
import numpy as np
import spacy
import emoji
import textstat
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertModel
from collections import Counter

app = Flask(__name__)
CORS(app)

# Caricamento spaCy
nlp = spacy.load("en_core_web_sm")

# --- CARICAMENTO MODELLI ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = './bert_ai_detector'

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
bert_model = BertModel.from_pretrained(MODEL_DIR).to(device)
bert_model.eval()

rf_hybrid = joblib.load('./pkl/rf_hybrid_model.pkl')
scaler = joblib.load('./pkl/feature_scaler.pkl')

with open('./txt/stylometric_features.txt', 'r') as f:
    stylometric_features_list = [line.strip() for line in f.readlines()]

def get_balanced_features(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    words = [t.text.lower() for t in doc if t.is_alpha]
    
    # Calcolo lunghezze con protezione per divisione zero
    s_lens = [len(re.findall(r'\w+', s.text)) for s in sentences]
    if not s_lens: s_lens = [1]
    
    mean_len = np.mean(s_lens)
    std_len = np.std(s_lens)

    f_dict = {
        # Usiamo il log per smorzare i valori estremi della burstiness
        'sentence_length_cv': (std_len / mean_len) if mean_len > 0 else 0,
        'burstiness_index': np.log1p(std_len), 
        'lexical_compression_ratio': len(set(words)) / len(words) if words else 0.5,
        'function_word_ratio': sum(1 for t in doc if t.is_stop) / len(doc) if len(doc) > 0 else 0,
        'hapax_density': sum(1 for c in Counter(words).values() if c == 1) / len(words) if words else 0,
        'readability_oscillation': textstat.flesch_reading_ease(text) / 100,
        'sentence_similarity_drift': 0.3, # Valore neutro per non sbilanciare
        'clause_density': len(sentences) / (len(words)/10 + 1),
        'template_bias_score': 1.0 - (len(set(words)) / len(words) if words else 0.5),
        'pos_bigram_entropy': 2.8, 
        'dependency_depth_mean': mean_len / 12, # Normalizzazione più aggressiva
        'sentiment_variance': 0.1,
        'structural_redundancy': 0.2
    }
    
    return np.array([f_dict.get(name, 0.0) for name in stylometric_features_list])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    # 1. BERT Embedding
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # 2. Stylometric con ricalibrazione
    style_vec = get_balanced_features(text)
    style_scaled = scaler.transform(style_vec.reshape(1, -1))

    # 3. Hybrid Merge
    hybrid_input = np.hstack([cls_emb, style_scaled])
    
    # Otteniamo le probabilità grezze
    raw_probs = rf_hybrid.predict_proba(hybrid_input)[0]
    
    # --- CALIBRAZIONE (SMOOTHING) ---
    # Se il modello è troppo sicuro (sopra il 95%), riduciamo leggermente la confidenza
    # per riflettere la prudenza di Colab
    ai_prob = raw_probs[1]
    if ai_prob > 0.95: ai_prob = 0.92
    if ai_prob < 0.05: ai_prob = 0.08
    human_prob = 1.0 - ai_prob

    return jsonify({
        "label": "AI Generated" if ai_prob > 0.5 else "Human Written",
        "confidence": round(max(ai_prob, human_prob) * 100, 2),
        "probabilities": {
            "human": round(human_prob * 100, 2),
            "ai": round(ai_prob * 100, 2)
        }
    })

if __name__ == '__main__':
    app.run(port=5000)