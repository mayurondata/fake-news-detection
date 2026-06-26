"""
predictor.py
────────────
Loads the trained Bi-LSTM model + tokenizer and exposes a clean
`predict(text)` interface used by both the Streamlit app and any
REST API wrapper.
"""

from __future__ import annotations
import re
import pickle
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

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
        _tf = tf
        _keras = pad_sequences
    return _keras


# ─────────────────────────────────────────────────────────────────────────────
# Default paths (relative to project root)
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(__file__))
# Prefer .keras (native format, version-stable) over legacy .h5
_keras_path = os.path.join(_ROOT, "models", "best_model.keras")
_h5_path = os.path.join(_ROOT, "models", "best_model.h5")
DEFAULT_MODEL = _keras_path if os.path.exists(_keras_path) else _h5_path
DEFAULT_TOK = os.path.join(_ROOT, "models", "tokenizer.pkl")
DEFAULT_CFG = os.path.join(_ROOT, "models", "config.json")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (must mirror training pipeline exactly)
# ─────────────────────────────────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words = set(ENGLISH_STOP_WORDS)


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
        self._model = None
        self._tokenizer = None
        self._max_len = 512

        self._model_path = model_path
        self._tok_path = tok_path
        self._cfg_path = cfg_path

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return

        pad_sequences = _get_keras()

        # Config
        self._cfg = {}
        if os.path.exists(self._cfg_path):
            with open(self._cfg_path) as f:
                self._cfg = json.load(f)
            self._max_len = self._cfg.get("MAX_LEN", 512)

        # Tokenizer
        with open(self._tok_path, "rb") as f:
            self._tokenizer = pickle.load(f)

        # Model — robust loader that handles Keras 2 vs Keras 3 format mismatch.
        # Models trained in Colab (TF 2.x / Keras 2) and loaded locally on Keras 3
        # fail with "Unknown layer: NotEqual" because mask_zero=True in Embedding
        # is serialised differently across versions.
        self._model = self._load_model_safe(self._model_path)

    def _load_model_safe(self, path: str):
        """
        Try multiple loading strategies in order of preference:
          1. Native load (works when versions match)
          2. custom_object_scope with NotEqual stub (Keras 2→3 mismatch fix)
          3. compile=False fallback (skips optimizer restore)
        After successful load, re-saves as .keras format for future runs.
        """
        import tensorflow as tf

        keras_path = path.replace(".h5", ".keras")

        # ── Strategy 0: prefer the re-saved .keras file if it already exists ──
        if os.path.exists(keras_path):
            try:
                return tf.keras.models.load_model(keras_path, compile=False)
            except Exception:
                pass

        # ── Strategy 1: plain load ────────────────────────────────────────────
        try:
            model = tf.keras.models.load_model(path, compile=False)
            self._resave(model, keras_path)
            return model
        except Exception as e1:
            pass

        # ── Strategy 2: custom_object_scope with NotEqual stub ────────────────
        # Keras 3 doesn't recognise the 'NotEqual' layer that Keras 2 used
        # internally to implement mask_zero=True on Embedding layers.
        try:
            class _NotEqual(tf.keras.layers.Layer):
                """Stub that satisfies the deserialiser; masking is re-applied at runtime."""

                def call(self, inputs):
                    return tf.not_equal(inputs[0], inputs[1])

            with tf.keras.utils.custom_object_scope({"NotEqual": _NotEqual}):
                model = tf.keras.models.load_model(path, compile=False)
            self._resave(model, keras_path)
            return model
        except Exception as e2:
            pass

        # ── Strategy 3: weights-only rebuild ─────────────────────────────────
        # Reconstruct the exact same architecture and just load the weights.
        try:
            model = self._rebuild_architecture()
            model.load_weights(path, by_name=True, skip_mismatch=True)
            self._resave(model, keras_path)
            return model
        except Exception as e3:
            raise RuntimeError(
                f"Could not load model from '{path}' using any strategy.\n"
                f"  Strategy 1: {e1}\n"
                f"  Strategy 2: {e2}\n"
                f"  Strategy 3: {e3}\n\n"
                "Fix: open Colab, load best_model.h5, then run:\n"
                "  model.save('best_model.keras')\n"
                "and copy best_model.keras into your models/ folder."
            )

    def _resave(self, model, keras_path: str) -> None:
        """Save in native .keras format — avoids this error on all future runs."""
        try:
            model.save(keras_path)
        except Exception:
            pass   # non-fatal — original .h5 still usable next time

    def _rebuild_architecture(self):
        """
        Mirror of train.py build_model() using stored config.
        Used as last-resort weight loading fallback.
        """
        import tensorflow as tf
        from tensorflow.keras.layers import (
            Embedding, Bidirectional, LSTM, Dense, Dropout,
            SpatialDropout1D, GlobalMaxPooling1D,
        )
        from tensorflow.keras.regularizers import l2

        hp = dict(
            VOCAB_SIZE=self._cfg.get("VOCAB_SIZE", 30000),
            MAX_LEN=self._max_len,
            EMBED_DIM=self._cfg.get("EMBED_DIM", 128),
            LSTM_UNITS=self._cfg.get("LSTM_UNITS", 128),
            SPATIAL_DROP=0.30, LSTM_DROP=0.30,
            LSTM_REC_DROP=0.20, DENSE_DROP1=0.40,
            DENSE_DROP2=0.30, L2_REG=1e-4,
        )

        inputs = tf.keras.Input(shape=(hp["MAX_LEN"],), dtype="int32")
        x = Embedding(hp["VOCAB_SIZE"], hp["EMBED_DIM"],
                      mask_zero=True)(inputs)
        x = SpatialDropout1D(hp["SPATIAL_DROP"])(x)
        x = Bidirectional(LSTM(hp["LSTM_UNITS"], return_sequences=True,
                               dropout=hp["LSTM_DROP"],
                               recurrent_dropout=hp["LSTM_REC_DROP"],
                               kernel_regularizer=l2(hp["L2_REG"])))(x)
        x = Bidirectional(LSTM(hp["LSTM_UNITS"] // 2, return_sequences=True,
                               dropout=hp["LSTM_DROP"],
                               recurrent_dropout=hp["LSTM_REC_DROP"]))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation="relu",
                  kernel_regularizer=l2(hp["L2_REG"]))(x)
        x = Dropout(hp["DENSE_DROP1"])(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(hp["DENSE_DROP2"])(x)
        outputs = Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, text: str) -> PredictionResult:
        """Run inference on a single article string."""
        self._load()
        pad_sequences = _get_keras()

        clean = preprocess(text)
        seq = self._tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=self._max_len,
                               padding="post", truncating="post")

        real_prob: float = float(self._model.predict(padded, verbose=0)[0][0])
        fake_prob: float = 1.0 - real_prob
        label = "REAL" if real_prob >= 0.5 else "FAKE"
        confidence = real_prob if label == "REAL" else fake_prob

        return PredictionResult(
            label=label,
            confidence=round(confidence, 4),
            real_prob=round(real_prob, 4),
            fake_prob=round(fake_prob, 4),
            word_count=len(text.split()),
            clean_token_count=len(clean.split()),
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Batch inference for multiple articles."""
        self._load()
        pad_sequences = _get_keras()

        cleans = [preprocess(t) for t in texts]
        seqs = self._tokenizer.texts_to_sequences(cleans)
        padded = pad_sequences(seqs, maxlen=self._max_len,
                               padding="post", truncating="post")

        probs = self._model.predict(padded, batch_size=32, verbose=0).flatten()

        results = []
        for i, (text, real_prob) in enumerate(zip(texts, probs)):
            real_prob = float(real_prob)
            fake_prob = 1.0 - real_prob
            label = "REAL" if real_prob >= 0.5 else "FAKE"
            confidence = real_prob if label == "REAL" else fake_prob
            results.append(PredictionResult(
                label=label,
                confidence=round(confidence, 4),
                real_prob=round(real_prob, 4),
                fake_prob=round(fake_prob, 4),
                word_count=len(texts[i].split()),
                clean_token_count=len(cleans[i].split()),
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
