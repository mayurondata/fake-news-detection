"""
train.py
────────
Standalone training script. Mirrors the notebook exactly but
is cleaner for CI / re-training pipelines.

Usage:
    python src/train.py --data_dir data/ --model_dir models/
"""

import argparse, os, re, pickle, json, logging
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    Bidirectional, SpatialDropout1D, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HP = dict(
    VOCAB_SIZE   = 30_000,
    MAX_LEN      = 512,
    EMBED_DIM    = 128,
    LSTM_UNITS   = 128,
    BATCH_SIZE   = 64,
    EPOCHS       = 10,
    LR           = 1e-3,
    SPATIAL_DROP = 0.30,
    LSTM_DROP    = 0.30,
    LSTM_REC_DROP= 0.20,
    DENSE_DROP1  = 0.40,
    DENSE_DROP2  = 0.30,
    L2_REG       = 1e-4,
)


# ── Preprocessing ─────────────────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>",                 "", text)
    text = re.sub(r"\[.*?\]",               "", text)
    text = re.sub(r"[^a-z\s]",             "", text)
    text = re.sub(r"\s+",                  " ", text).strip()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in _stop_words and len(t) > 2
    ]
    return " ".join(tokens)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_isot(data_dir: str) -> pd.DataFrame:
    """Load the ISOT True/Fake CSVs."""
    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"Expected True.csv and Fake.csv in '{data_dir}'.\n"
            "Download from: https://www.kaggle.com/datasets/emineyetm/fake-and-real-news-dataset"
        )

    true_df = pd.read_csv(true_path);  true_df["label"] = 1
    fake_df = pd.read_csv(fake_path);  fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["content", "label"]].sample(frac=1, random_state=SEED).reset_index(drop=True)

    log.info(f"Loaded {len(df):,} articles  (real={true_df.shape[0]:,}, fake={fake_df.shape[0]:,})")
    return df


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(hp: dict) -> tf.keras.Model:
    # Use Functional API with explicit Input layer — fixes the "0 params / unbuilt"
    # issue in TF 2.x where Sequential + input_length on Embedding no longer
    # triggers automatic shape inference in newer versions.
    inputs = tf.keras.Input(shape=(hp["MAX_LEN"],), dtype="int32", name="token_ids")

    x = Embedding(hp["VOCAB_SIZE"], hp["EMBED_DIM"], mask_zero=True, name="embedding")(inputs)
    x = SpatialDropout1D(hp["SPATIAL_DROP"])(x)

    x = Bidirectional(LSTM(hp["LSTM_UNITS"], return_sequences=True,
                           dropout=hp["LSTM_DROP"],
                           recurrent_dropout=hp["LSTM_REC_DROP"],
                           kernel_regularizer=l2(hp["L2_REG"])),
                      name="bilstm_1")(x)
    x = Bidirectional(LSTM(hp["LSTM_UNITS"] // 2, return_sequences=True,
                           dropout=hp["LSTM_DROP"],
                           recurrent_dropout=hp["LSTM_REC_DROP"]),
                      name="bilstm_2")(x)

    x = GlobalMaxPooling1D()(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(hp["L2_REG"]))(x)
    x = Dropout(hp["DENSE_DROP1"])(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(hp["DENSE_DROP2"])(x)
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BiLSTM_FakeNews")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp["LR"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    return model


# ── Main training pipeline ────────────────────────────────────────────────────
def train(data_dir: str, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load & preprocess
    df = load_isot(data_dir)
    log.info("Preprocessing text …")
    df["clean"] = df["content"].apply(preprocess)

    # 2. Split
    # IMPORTANT: keep X as a plain Python list — wrapping 44K long strings in
    # np.array() forces numpy to pad every string to the length of the longest
    # one (U37834), blowing up to 4-5 GiB. Lists have no such overhead.
    # Only y (small ints) becomes a numpy array.
    X = df["clean"].tolist()
    y = np.array(df["label"].tolist(), dtype=np.int32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=SEED, stratify=y_tr)
    log.info(f"Split → train:{len(X_tr):,}  val:{len(X_va):,}  test:{len(X_te):,}")

    # 3. Tokenize
    tok = Tokenizer(num_words=HP["VOCAB_SIZE"], oov_token="<OOV>")
    tok.fit_on_texts(X_tr)

    def encode(texts):
        return pad_sequences(tok.texts_to_sequences(texts),
                             maxlen=HP["MAX_LEN"], padding="post", truncating="post")

    X_tr_p, X_va_p, X_te_p = encode(X_tr), encode(X_va), encode(X_te)

    # Save tokenizer
    tok_path = os.path.join(model_dir, "tokenizer.pkl")
    with open(tok_path, "wb") as f:
        pickle.dump(tok, f)
    log.info(f"Tokenizer saved → {tok_path}")

    # 4. Build model
    model = build_model(HP)
    model.summary(print_fn=log.info)

    # 5. Callbacks
    best_path = os.path.join(model_dir, "best_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_auc", patience=3, restore_best_weights=True,
                      mode="max", verbose=1),
        ModelCheckpoint(best_path, monitor="val_auc", save_best_only=True,
                        mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                          min_lr=1e-6, verbose=1),
    ]

    # 6. Train
    log.info("Training …")
    model.fit(
        X_tr_p, y_tr,
        validation_data=(X_va_p, y_va),
        epochs=HP["EPOCHS"],
        batch_size=HP["BATCH_SIZE"],
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluate
    log.info("Evaluating on test set …")
    y_prob = model.predict(X_te_p, batch_size=128, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_te, y_prob)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(classification_report(y_te, y_pred, target_names=["Fake", "Real"]))
    print(f"ROC-AUC: {auc:.4f}")

    # 8. Save config
    cfg = {**HP, "test_accuracy": float((y_pred == y_te).mean()), "test_auc": float(auc)}
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    log.info(f"Config saved → {cfg_path}")
    log.info(f"Best model  → {best_path}")
    log.info(f"✅  Training complete  |  Test Acc: {cfg['test_accuracy']*100:.2f}%  |  AUC: {auc:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bi-LSTM Fake News Detector")
    parser.add_argument("--data_dir",  default="data/",   help="Folder with True.csv & Fake.csv")
    parser.add_argument("--model_dir", default="models/", help="Where to save model artifacts")
    args = parser.parse_args()
    train(args.data_dir, args.model_dir)
