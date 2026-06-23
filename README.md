# 📰 Fake News Detector — Bidirectional LSTM

> **End-to-end NLP deep learning project** — from raw data to a live deployed web app.  
> Classifies news articles as **REAL** or **FAKE** using a stacked Bidirectional LSTM trained on the ISOT dataset.

[![CI](https://github.com/yourusername/fake-news-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/fake-news-detector/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **~99%** |
| ROC-AUC | **~0.999** |
| F1 (Fake) | **~0.99** |
| F1 (Real) | **~0.99** |

---

## 🏗 Project Structure

```
fake-news-detector/
├── data/                        # Raw CSVs (not committed — see Setup)
│   ├── True.csv
│   └── Fake.csv
│
├── models/                      # Saved artifacts (after training)
│   ├── best_model.h5            # Trained Bi-LSTM weights
│   ├── tokenizer.pkl            # Fitted Keras tokenizer
│   └── config.json              # Hyperparams + test metrics
│
├── notebooks/
│   └── 01_EDA_and_Training.ipynb   # Full walkthrough: EDA → model → eval
│
├── src/
│   ├── train.py                 # Standalone training script (CLI)
│   └── predictor.py             # Inference class used by the app
│
├── streamlit_app/
│   └── app.py                   # 🚀 Streamlit web app
│
├── tests/
│   └── test_predictor.py        # Pytest unit + integration tests
│
├── .github/workflows/ci.yml     # GitHub Actions CI pipeline
├── .streamlit/config.toml       # Streamlit theme config
├── requirements.txt
└── README.md
```

---

## ⚙️ Model Architecture

```
Input text
    │
    ▼
Text Preprocessing     → lowercase, URL/HTML removal, lemmatization, stopword removal
    │
    ▼
Keras Tokenizer        → 30,000 vocab, integer-encoded + padded to 512 tokens
    │
    ▼
Embedding Layer        → 128-dim trainable dense vectors
SpatialDropout1D(0.3)
    │
    ▼
BiLSTM Layer 1         → 128 units × 2 directions (return_sequences=True)
                          dropout=0.3, recurrent_dropout=0.2, L2 regularisation
    │
    ▼
BiLSTM Layer 2         → 64 units × 2 directions (return_sequences=True)
    │
    ▼
GlobalMaxPooling1D     → most salient feature per dimension
    │
    ▼
Dense(128) → Dropout(0.4) → Dense(64) → Dropout(0.3)
    │
    ▼
Dense(1, sigmoid)      → P(real) ∈ [0, 1]
```

**Why Bidirectional LSTM?**  
News articles contain long-range dependencies — a clue at word 300 may depend on context at word 10. Bi-LSTMs read sequences in both directions simultaneously, capturing richer contextual signals than unidirectional RNNs. The stacked architecture learns hierarchical features, while GlobalMaxPooling makes the model robust to position.

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset

Download [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-and-real-news-dataset) from Kaggle.  
Place `True.csv` and `Fake.csv` inside the `data/` folder.

```
data/
├── True.csv    (~21,000 articles)
└── Fake.csv    (~23,500 articles)
```

### 3a. Train via CLI (recommended)

```bash
python src/train.py --data_dir data/ --model_dir models/
```

Expected output:
```
2024-01-01 12:00:00  INFO  Loaded 44,898 articles  (real=21,417, fake=23,481)
2024-01-01 12:00:05  INFO  Preprocessing text …
...
TEST RESULTS
============================================================
              precision    recall  f1-score   support

        Fake       0.99      0.99      0.99      3350
        Real       0.99      0.99      0.99      3085

    accuracy                           0.99      6435
ROC-AUC: 0.9993
```

### 3b. Train via Jupyter Notebook

```bash
jupyter notebook notebooks/01_EDA_and_Training.ipynb
```

The notebook includes full EDA (word clouds, distributions, length analysis) before training.

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run tests

```bash
pytest tests/ -v --cov=src
```

---

## ☁️ Deployment (Streamlit Cloud — Free)

1. Push this repo to GitHub (ensure `models/` is committed or use Git LFS)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and set **Main file path** to `streamlit_app/app.py`
4. Click **Deploy** — live in ~2 minutes ✅

> **Tip:** For large model files (>100MB), use [Git LFS](https://git-lfs.github.com) or load the model from an S3/GCS bucket via environment variables.

---

## 📊 Dataset

**ISOT Fake News Dataset** — University of Victoria, 2018  
- ~21,400 real articles (Reuters)  
- ~23,500 fake articles (flagged by PolitiFact & other fact-checkers)  
- Domains: politics, world news, US news

Alternative dataset: [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) (~72K articles, more diverse sources)

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | TensorFlow 2 / Keras |
| NLP Preprocessing | NLTK (stopwords, lemmatizer) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Testing | Pytest + pytest-cov |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Cloud (free tier) |

---

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 30,000 |
| Max sequence length | 512 tokens |
| Embedding dimension | 128 |
| LSTM units | 128 (layer 1), 64 (layer 2) |
| Batch size | 64 |
| Optimizer | Adam (lr=1e-3) |
| LR schedule | ReduceLROnPlateau (factor=0.5) |
| Regularisation | SpatialDropout, L2, EarlyStopping |
| Early stopping | Monitors val_AUC (patience=3) |

---

## ⚠️ Limitations & Future Work

- **Dataset bias:** ISOT articles are heavily political; performance on other domains (health, science misinformation) may be lower
- **Source leak:** Real articles are all from Reuters — the model may partly learn source style rather than content alone
- **No transformer baseline:** Comparing against DistilBERT / RoBERTa would be a natural next step
- **Explainability:** LIME / SHAP integration for word-level attribution is planned

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- Ahmed H, Traore I, Saad S. — *"Detecting opinion spams and fake news using text classification"* (2018)  
- [ISOT Research Lab](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php), University of Victoria  
- Streamlit team for the free deployment platform
