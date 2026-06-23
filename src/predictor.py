"""
predictor.py
────────────
Loads the trained Bi-LSTM model + tokenizer and exposes a clean
`predict(text)` interface used by both the Streamlit app and any
REST API wrapper.
"""

from __future__ import annotations
import re, pickle, json, os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports for heavy libs so the module is importable without TF installed
# ─────────────────────────────────────────────────────────────────────────────
_tf = None
_keras = None

def _get_keras():
    global _tf, _keras
    if _keras is None:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        _tf     = tf
        _keras  = pad_sequences
    return _keras


# ─────────────────────────────────────────────────────────────────────────────
# Default paths (relative to project root)
# ─────────────────────────────────────────────────────────────────────────────
_ROOT          = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL  = os.path.join(_ROOT, "models", "best_model.h5")
DEFAULT_TOK    = os.path.join(_ROOT, "models", "tokenizer.pkl")
DEFAULT_CFG    = os.path.join(_ROOT, "models", "config.json")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (must mirror training pipeline exactly)
# ─────────────────────────────────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Clean and tokenise a raw news article string."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)   # URLs
    text = re.sub(r"<.*?>",                 "", text)   # HTML tags
    text = re.sub(r"\[.*?\]",               "", text)   # brackets
    text = re.sub(r"[^a-z\s]",             "", text)   # keep letters only
    text = re.sub(r"\s+",                  " ", text).strip()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in _stop_words and len(t) > 2
    ]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    label:       str          # "REAL" or "FAKE"
    confidence:  float        # 0.0 – 1.0  (prob of the predicted class)
    real_prob:   float        # raw sigmoid output
    fake_prob:   float
    word_count:  int
    clean_token_count: int


# ─────────────────────────────────────────────────────────────────────────────
# Main predictor class
# ─────────────────────────────────────────────────────────────────────────────
class FakeNewsPredictor:
    """
    Wrapper around the trained Bi-LSTM model.

    Usage
    -----
    >>> predictor = FakeNewsPredictor()
    >>> result    = predictor.predict("Breaking: Scientists discover ...")
    >>> print(result.label, result.confidence)
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        tok_path:   str = DEFAULT_TOK,
        cfg_path:   str = DEFAULT_CFG,
    ) -> None:
        self._model     = None
        self._tokenizer = None
        self._max_len   = 512

        self._model_path = model_path
        self._tok_path   = tok_path
        self._cfg_path   = cfg_path

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return

        pad_sequences = _get_keras()

        # Config
        if os.path.exists(self._cfg_path):
            with open(self._cfg_path) as f:
                cfg = json.load(f)
            self._max_len = cfg.get("MAX_LEN", 512)

        # Tokenizer
        with open(self._tok_path, "rb") as f:
            self._tokenizer = pickle.load(f)

        # Model
        self._model = _tf.keras.models.load_model(self._model_path)

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, text: str) -> PredictionResult:
        """Run inference on a single article string."""
        self._load()
        pad_sequences = _get_keras()

        clean        = preprocess(text)
        seq          = self._tokenizer.texts_to_sequences([clean])
        padded       = pad_sequences(seq, maxlen=self._max_len,
                                     padding="post", truncating="post")

        real_prob: float = float(self._model.predict(padded, verbose=0)[0][0])
        fake_prob: float = 1.0 - real_prob
        label            = "REAL" if real_prob >= 0.5 else "FAKE"
        confidence       = real_prob if label == "REAL" else fake_prob

        return PredictionResult(
            label             = label,
            confidence        = round(confidence, 4),
            real_prob         = round(real_prob, 4),
            fake_prob         = round(fake_prob, 4),
            word_count        = len(text.split()),
            clean_token_count = len(clean.split()),
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Batch inference for multiple articles."""
        self._load()
        pad_sequences = _get_keras()

        cleans   = [preprocess(t) for t in texts]
        seqs     = self._tokenizer.texts_to_sequences(cleans)
        padded   = pad_sequences(seqs, maxlen=self._max_len,
                                 padding="post", truncating="post")

        probs = self._model.predict(padded, batch_size=32, verbose=0).flatten()

        results = []
        for i, (text, real_prob) in enumerate(zip(texts, probs)):
            real_prob  = float(real_prob)
            fake_prob  = 1.0 - real_prob
            label      = "REAL" if real_prob >= 0.5 else "FAKE"
            confidence = real_prob if label == "REAL" else fake_prob
            results.append(PredictionResult(
                label             = label,
                confidence        = round(confidence, 4),
                real_prob         = round(real_prob, 4),
                fake_prob         = round(fake_prob, 4),
                word_count        = len(texts[i].split()),
                clean_token_count = len(cleans[i].split()),
            ))

        return results

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Singleton — avoids reloading model on every Streamlit re-run
_instance: Optional[FakeNewsPredictor] = None

def get_predictor() -> FakeNewsPredictor:
    global _instance
    if _instance is None:
        _instance = FakeNewsPredictor()
    return _instance
