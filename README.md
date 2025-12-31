# 🧠 AI Detector – Hybrid BERT & Stylometric Analysis

## 📌 Overview
Questo progetto implementa un **sistema end-to-end per il rilevamento di testi generati da Intelligenza Artificiale**, combinando modelli NLP moderni e tecniche di analisi stilometrica.  
L’obiettivo è distinguere testi **scritti da esseri umani** da testi **generati da modelli AI**, adottando un approccio sia **predittivo** sia **interpretabile**.

Il sistema è stato progettato come applicazione **full-stack containerizzata**, includendo backend AI, frontend web e orchestrazione tramite Docker.

---

## 🏗️ Architettura del Sistema

┌────────────┐ REST API ┌──────────────┐
│ React UI │ ───────────────▶ │ FastAPI ML │
│ Frontend │ │ Backend │
└────────────┘ └──────┬───────┘
│
┌─────────▼─────────┐
│ Modelli ML │
│ BERT (.pth) │
│ Signature (.pkl) │
└───────────────────┘


- **Frontend**: React + TailwindCSS  
- **Backend**: Python + FastAPI  
- **Modelli**: BERT + feature stilometriche  
- **Deployment**: Docker & Docker Compose  

---

## 📁 Struttura del Progetto

bert_ai_detector/
│
├── app.py # Backend FastAPI
├── requirements.txt # Dipendenze Python
├── Dockerfile # Backend container
│
├── react-app/ # Frontend React
│ ├── Dockerfile
│ ├── src/
│ │ └── components/
│ │ └── AIDetectorInterface.jsx
│ └── ...
│
├── pth/ # Modelli deep learning
│ └── best_bert.pth
│
├── pkl/ # Modelli ML / signature
├── txt/ # File di supporto
│
├── *.ipynb # Notebook (EDA, training, analisi)
├── *.csv # Dataset
│
├── docker-compose.yml
└── README.md


I notebook Jupyter sono volutamente esclusi dai container per separare la fase di **training** da quella di **inference**.

---

## 🔬 Metodologia

### 1️⃣ Exploratory Data Analysis (EDA)
- Analisi statistica dei testi AI vs Human
- Studio di lunghezza, variabilità e struttura
- Supporto alle decisioni di feature engineering

### 2️⃣ Signature Stilometriche
- Diversità lessicale
- Ripetitività strutturale
- Burstiness
- Lunghezza media delle frasi

Queste feature forniscono un livello di **interpretabilità** complementare ai modelli deep learning.

### 3️⃣ Modello Ibrido (BERT + Feature Stilometriche)
- Embedding contestuali ottenuti tramite BERT
- Integrazione con feature linguistiche manuali
- Migliore robustezza e generalizzazione rispetto ad approcci singoli

---

## ⚙️ Backend – FastAPI

Il backend espone un’API REST per l’analisi dei testi.

### Endpoint
`POST /analyze`

### Input
```json
{
  "text": "Testo da analizzare"
}

## Output
{
  "isAI": true,
  "confidence": 87.3,
  "metrics": {
    "lexical_diversity": 0.42,
    "burstiness": 3.1,
    "avg_sentence_length": 18.7
  }
}

## 🎨 Frontend – React App

Il frontend fornisce un’interfaccia web interattiva per:

inserimento del testo

validazione dell’input

visualizzazione dei risultati e delle metriche

comunicazione diretta con il backend tramite REST API

## 🐳 Containerizzazione con Docker

Il sistema è completamente containerizzato tramite Docker Compose, che orchestra:

backend AI (FastAPI + modelli ML)

frontend React

Avvio del progetto
docker-compose build
docker-compose up

| Servizio |  | Descrizione     |
| -------- |  | --------------- |
| Backend  |  | API AI Detector |
| Frontend |  | Interfaccia Web |

La comunicazione tra frontend e backend avviene tramite service name Docker, garantendo portabilità e riproducibilità.

 ## 🎓 Scelte Progettuali

Separazione tra training e inference

Approccio ibrido per bilanciare performance e interpretabilità

Containerizzazione per:

riproducibilità degli esperimenti

isolamento dell’ambiente

semplicità di deploy

Interfaccia grafica come strumento di analisi e non solo demo

## ⚠️ Limiti e Sviluppi Futuri

Generalizzazione rispetto a modelli AI futuri

Integrazione del supporto GPU (CUDA)

Valutazione cross-domain

Logging e monitoring delle predizioni

Supporto per analisi batch

## 👤 Autore

Progetto sviluppato come lavoro accademico nell’ambito di Machine Learning e Natural Language Processing.

## 🏁 Conclusione

Il progetto dimostra come sia possibile costruire un AI Detector moderno e completo, combinando:

analisi statistica

modelli deep learning

interpretabilità linguistica

ingegneria del software

Il risultato è un sistema modulare, riproducibile e pronto al deploy.
