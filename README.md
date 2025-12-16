# ProgettoFinale_RosyAnnaAnto

## In questo progetto utilizzeremo un dataset contenente testi etichettati come AI-generated o Human-written, insieme a diverse caratteristiche linguistiche e semantiche.

## 🎯 Obiettivi del progetto
Al termine del progetto sarai in grado di:

- Analizzare le **differenze linguistiche** tra testi umani e generati da AI
- Svolgere un’analisi esplorativa dei dati (EDA)
- Costruire un **modello di Machine Learning / NLP** per la classificazione
- Valutare e interpretare le prestazioni del modello
- Comunicare risultati e insight in modo chiaro e strutturato

## 📂 Dataset
Il dataset contiene **500 osservazioni** con le seguenti colonne principali:

| Colonna | Descrizione |
|-------|-------------|
| `label` | Classe target: `human` / `ai` |
| `text` | Contenuto testuale |
| `topic` | Argomento del testo |
| `length_chars` | Lunghezza in caratteri |
| `length_words` | Lunghezza in parole |
| `sentiment` | Sentiment del testo |
| `quality_score` | Valutazione qualitativa |
| `plagiarism_score` | Similarità con altri testi |
| `source_detail` | Origine del testo |
| `timestamp` | Data e ora |
| `notes` | Note aggiuntive |

---

## 🧩 Fasi del progetto

### 1️⃣ Data Understanding & Exploratory Data Analysis (EDA)
- Analisi della distribuzione delle classi
- Studio dei topic più frequenti
- Confronto tra testi AI e Human rispetto a:
  - Lunghezza
  - Sentiment
  - Quality score
  - Plagiarism score
- Visualizzazioni consigliate:
  - Istogrammi
  - Boxplot
  - Grafici a barre
  - Heatmap di correlazione

---

### 2️⃣ Preprocessing dei dati
- Gestione dei valori mancanti
- Pulizia del testo (lowercase, rimozione punteggiatura, stopwords, ecc.)
- Encoding delle variabili categoriche
- Feature engineering, ad esempio:
  - Rapporto parole/caratteri
  - Valore assoluto del sentiment
  - Indicatori di stile linguistico

---

### 3️⃣ Feature Engineering NLP 
- Bag of Words
- TF-IDF
- N-grams
- Confronto tra:
  - Feature strutturate
  - Feature testuali
  - Approccio ibrido

---

### 4️⃣ Modellazione
Costruisci e confronta **almeno due modelli**, ad esempio:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- (Bonus) Modelli NLP avanzati

**Target:** `label`

---

### 5️⃣ Valutazione del modello
- Suddivisione Train / Test
- Metriche di valutazione:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Analisi degli errori:
  - False Positive
  - False Negative

---
### 6️⃣ Interpretabilità & Insight
- Analisi delle feature più importanti
- Quali caratteristiche rendono un testo più simile a uno generato da AI?
- Limiti del modello
- Bias e limiti del dataset

---

### 7️⃣ Conclusioni e sviluppi futuri
- Sintesi dei risultati ottenuti
- Possibili miglioramenti:
  - Dataset più ampio
  - Feature linguistiche più avanzate
  - Modelli deep learning
- Possibili applicazioni reali:
  - Scuole e università
  - Piattaforme editoriali
  - Content moderation

---
## 📦 Deliverable richiesti
- Notebook Python ben documentato
- Presentazione finale (10–15 slide)
- (Opzionale) Demo o dashboard

---

## ⭐ Criteri di valutazione
- Chiarezza e completezza dell’analisi
- Correttezza tecnica
- Qualità delle visualizzazioni
- Capacità di interpretazione dei risultati
- Qualità della comunicazione finale

---