"""
app.py  ─  Fake News Detector  ·  Streamlit UI
────────────────────────────────────────────────────
Run locally:   streamlit run streamlit_app/app.py
Deploy:        push to GitHub → connect on share.streamlit.io
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.predictor import get_predictor, preprocess

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  — dark editorial theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --green:    #2ecc71;
    --red:      #e74c3c;
    --blue:     #58a6ff;
    --yellow:   #f0c14b;
    --muted:    #8b949e;
    --text:     #e6edf3;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background: var(--bg) !important; }

[data-testid="stSidebar"]  { background: var(--surface) !important; border-right: 1px solid var(--border); }

p, li, label, .stMarkdown { color: var(--text) !important; font-family: 'Inter', sans-serif; }

/* ── Hero header ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 5vw, 3.6rem);
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.1;
    color: var(--text);
    margin: 0;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: var(--muted);
    margin-top: .5rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f6feb33, #58a6ff22);
    border: 1px solid #1f6feb;
    color: var(--blue);
    font-size: .7rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: .8rem;
    letter-spacing: .08em;
}

/* ── Verdict card ── */
.verdict-real {
    background: linear-gradient(135deg, #0d2818, #0d2818);
    border: 2px solid var(--green);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.verdict-fake {
    background: linear-gradient(135deg, #2d0d0d, #2d0d0d);
    border: 2px solid var(--red);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.verdict-label {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 4px;
}
.verdict-conf {
    font-family: 'Inter', sans-serif;
    font-size: .9rem;
    color: var(--muted);
    margin-top: .3rem;
}

/* ── Metric card ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--blue);
}
.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: .8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .08em;
}

/* ── Textarea ── */
textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .95rem !important;
}
textarea:focus { border-color: var(--blue) !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: .7rem 2.5rem !important;
    font-size: 1rem !important;
    transition: opacity .2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── History table ── */
.history-row {
    display: flex;
    align-items: center;
    gap: .8rem;
    padding: .7rem 1rem;
    border-bottom: 1px solid var(--border);
    font-family: 'Inter', sans-serif;
    font-size: .85rem;
    color: var(--text);
}
.pill-real { background:#0d2818; color:var(--green); border:1px solid var(--green); border-radius:20px; padding:2px 10px; font-size:.75rem; }
.pill-fake { background:#2d0d0d; color:var(--red);   border:1px solid var(--red);   border-radius:20px; padding:2px 10px; font-size:.75rem; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Playfair Display",serif; font-size:1.4rem;
                font-weight:900; color:#e6edf3; margin-bottom:.3rem;'>
        🔍 Fake News Detector
    </div>
    <div style='color:#8b949e; font-size:.82rem; font-family:Inter,sans-serif;'>
        Bidirectional LSTM · NLP · Deep Learning
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### ⚙️ Model Info")
    st.markdown("""
    <div style='font-family:Inter,sans-serif; font-size:.83rem; color:#8b949e; line-height:1.8;'>
    <b style='color:#e6edf3;'>Architecture</b> — Bi-LSTM × 2 layers<br>
    <b style='color:#e6edf3;'>Embedding</b> — 128-dim, 30K vocab<br>
    <b style='color:#e6edf3;'>Dataset</b> — ISOT Fake News (~45K)<br>
    <b style='color:#e6edf3;'>Test Accuracy</b> — ~99%<br>
    <b style='color:#e6edf3;'>Test AUC</b> — ~0.999
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📚 Resources")
    st.markdown("""
    <div style='font-family:Inter,sans-serif; font-size:.83rem; line-height:2;'>
    <a href='https://github.com/yourusername/fake-news-detector'
       style='color:#58a6ff;'>📂 GitHub Repo</a><br>
    <a href='https://www.kaggle.com/datasets/emineyetm/fake-and-real-news-dataset'
       style='color:#58a6ff;'>📊 ISOT Dataset</a><br>
    <a href='https://arxiv.org/abs/2005.00033'
       style='color:#58a6ff;'>📄 Research Paper</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Demo article switcher
    st.markdown("#### 🧪 Try an Example")
    example_choice = st.selectbox(
        "Load a sample article:",
        ["— choose —", "Real news sample", "Fake news sample"],
        label_visibility="collapsed"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sample articles
# ─────────────────────────────────────────────────────────────────────────────
REAL_SAMPLE = """
The Federal Reserve raised interest rates by a quarter percentage point on Wednesday,
continuing its campaign to bring down inflation while signaling it may be nearing the
end of its rate-hiking cycle. Fed Chair Jerome Powell said at a news conference that
officials have seen encouraging progress on inflation but are not yet confident it is
on a sustainable path back to the 2% target. The decision was unanimous among the
12 voting members of the Federal Open Market Committee, which sets monetary policy.
""".strip()

FAKE_SAMPLE = """
BREAKING: Scientists at a top secret underground laboratory have confirmed the existence
of a massive cover-up involving world leaders and extraterrestrial beings. According to
anonymous insider sources, the government has been hiding alien technology for decades
that could solve all energy problems. Documents leaked from a whistleblower prove that
the deep state has been suppressing this information to maintain control over the
global population. Share this before it gets taken down!
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Load example
if example_choice == "Real news sample":
    st.session_state.input_text = REAL_SAMPLE
elif example_choice == "Fake news sample":
    st.session_state.input_text = FAKE_SAMPLE


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 2.5rem 0 1.5rem;'>
  <div class='hero-badge'>Bi-LSTM · NLP · Deep Learning</div>
  <div class='hero-title'>Fake News<br>Detector</div>
  <div class='hero-sub'>
    Paste a news article below — our deep learning model classifies it in milliseconds.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1.15, 1], gap="large")

with col_input:
    st.markdown("#### 📰 Article Text")
    article_text = st.text_area(
        label="article_input",
        label_visibility="collapsed",
        value=st.session_state.input_text,
        height=280,
        placeholder="Paste a news headline + body text here...\n\n"
                    "Tip: more text = more accurate prediction.",
        key="article_input_widget"
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        run_btn = st.button("🔍 Analyse Article", use_container_width=True)
    with col_clear:
        if st.button("✕ Clear", use_container_width=True):
            st.session_state.input_text = ""
            st.rerun()

    # Token preview
    if article_text.strip():
        clean = preprocess(article_text)
        n_words  = len(article_text.split())
        n_tokens = len(clean.split())
        st.markdown(
            f"<div style='color:#8b949e; font-size:.8rem; font-family:Inter,sans-serif;"
            f"margin-top:.5rem;'>"
            f"📝 {n_words} words → "
            f"<span style='color:#58a6ff;'>{n_tokens} tokens</span> after preprocessing"
            f"</div>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
with col_result:
    st.markdown("#### 🎯 Verdict")

    if run_btn:
        if not article_text.strip():
            st.warning("⚠️ Please enter some article text first.")
        elif len(article_text.split()) < 10:
            st.warning("⚠️ Article too short — add at least 10 words for a reliable prediction.")
        else:
            with st.spinner("Loading model & analysing..."):
                predictor = get_predictor()
                t0        = time.time()
                result    = predictor.predict(article_text)
                elapsed   = time.time() - t0

            # ── Verdict card ──────────────────────────────────────────────────
            icon    = "✅" if result.label == "REAL" else "🚨"
            cls     = "verdict-real" if result.label == "REAL" else "verdict-fake"
            color   = "#2ecc71" if result.label == "REAL" else "#e74c3c"
            conf_pct = f"{result.confidence * 100:.1f}%"

            st.markdown(f"""
            <div class='{cls}'>
                <div class='verdict-label' style='color:{color};'>{icon} {result.label}</div>
                <div class='verdict-conf'>Confidence: <b style='color:{color};'>{conf_pct}</b></div>
                <div class='verdict-conf' style='margin-top:.2rem;'>Inference: {elapsed*1000:.0f}ms</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Probability gauge ─────────────────────────────────────────────
            fig = go.Figure(go.Indicator(
                mode   = "gauge+number",
                value  = result.real_prob * 100,
                title  = {"text": "Real News Probability", "font": {"color": "#e6edf3", "size": 14}},
                number = {"suffix": "%", "font": {"color": "#e6edf3", "size": 28}},
                gauge  = {
                    "axis"      : {"range": [0, 100], "tickcolor": "#8b949e",
                                   "tickfont": {"color": "#8b949e"}},
                    "bar"       : {"color": "#2ecc71" if result.real_prob >= 0.5 else "#e74c3c"},
                    "bgcolor"   : "#161b22",
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0,  40], "color": "#3d1515"},
                        {"range": [40, 60], "color": "#2a2a15"},
                        {"range": [60, 100],"color": "#153d20"},
                    ],
                    "threshold": {
                        "line": {"color": "#f0c14b", "width": 3},
                        "thickness": 0.75, "value": 50
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor = "#0d1117",
                font          = {"color": "#e6edf3"},
                height        = 220,
                margin        = dict(l=20, r=20, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Mini metrics row ──────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-value'>{result.real_prob*100:.0f}%</div>
                  <div class='metric-label'>Real prob</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-value'>{result.fake_prob*100:.0f}%</div>
                  <div class='metric-label'>Fake prob</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-value'>{result.word_count}</div>
                  <div class='metric-label'>Words</div>
                </div>""", unsafe_allow_html=True)

            # ── Add to history ────────────────────────────────────────────────
            st.session_state.history.insert(0, {
                "snippet":    article_text[:70] + "...",
                "label":      result.label,
                "confidence": conf_pct,
            })
            if len(st.session_state.history) > 10:
                st.session_state.history = st.session_state.history[:10]

    else:
        # Placeholder
        st.markdown("""
        <div style='border:1px dashed #30363d; border-radius:12px; padding:3rem 2rem;
                    text-align:center; color:#8b949e; font-family:Inter,sans-serif;'>
            <div style='font-size:2.5rem; margin-bottom:.8rem;'>🔍</div>
            <div style='font-size:.95rem;'>
                Paste an article and click<br>
                <b style='color:#e6edf3;'>Analyse Article</b> to get a verdict.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# History panel
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕓 Recent Predictions")

    for i, h in enumerate(st.session_state.history):
        pill_cls   = "pill-real" if h["label"] == "REAL" else "pill-fake"
        icon_small = "✅" if h["label"] == "REAL" else "🚨"
        st.markdown(f"""
        <div class='history-row'>
            <span class='{pill_cls}'>{icon_small} {h['label']}</span>
            <span style='color:#8b949e;'>{h['confidence']}</span>
            <span style='flex:1; overflow:hidden; text-overflow:ellipsis;
                         white-space:nowrap;'>{h['snippet']}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑 Clear history"):
        st.session_state.history = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture explainer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🧠 How it works — Model Architecture", expanded=False):
    arch_col, desc_col = st.columns([1, 1.2], gap="large")
    with arch_col:
        st.markdown("""
        ```
        Input: raw text string
              │
              ▼
        ┌─────────────────────┐
        │  Text Preprocessing │  lowercase, remove noise,
        │  + Lemmatization    │  stopword removal
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Tokenizer          │  30K vocab → integer IDs
        │  + Padding (512)    │  fixed-length sequences
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Embedding Layer    │  128-dim dense vectors
        │  + SpatialDropout   │
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Bi-LSTM Layer 1    │  128 units × 2 directions
        │  (return_sequences) │  captures long-range context
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Bi-LSTM Layer 2    │  64 units × 2 directions
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  GlobalMaxPooling   │  most salient feature per dim
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Dense 128 → 64     │  + Dropout regularisation
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Dense 1 (sigmoid)  │  P(real) ∈ [0, 1]
        └─────────────────────┘
        ```
        """)
    with desc_col:
        st.markdown("""
        **Why Bidirectional LSTM?**

        News articles have long-range linguistic patterns — a word at position 300
        may contextually depend on a word at position 10. Standard RNNs struggle;
        Bi-LSTMs read the sequence *forward and backward simultaneously*, capturing
        richer context.

        **Key design choices**

        | Choice | Rationale |
        |--------|-----------|
        | Bi-LSTM × 2 | Stacked LSTMs learn hierarchical features |
        | GlobalMaxPooling | Picks the single most activated feature per dim — robust to position |
        | SpatialDropout | Drops entire embedding dimensions, more effective than standard dropout for sequences |
        | L2 regularisation | Prevents overfit on dataset quirks (news source patterns) |
        | EarlyStopping on AUC | Monitors calibration, not just accuracy |

        **Training details**
        - Dataset: ISOT Fake News (~21K real + ~24K fake articles)
        - Optimiser: Adam (lr=1e-3, ReduceLROnPlateau)
        - Batch size: 64 · Max seq len: 512
        - Best val AUC achieved: **0.999+**
        """)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#8b949e; font-size:.8rem; font-family:Inter,sans-serif;
            padding: 2rem 0 1rem; border-top: 1px solid #30363d; margin-top: 2rem;'>
  Built with TensorFlow · Streamlit · Plotly &nbsp;|&nbsp;
  Bi-LSTM Fake News Detector &nbsp;|&nbsp;
  <a href='https://github.com/yourusername/fake-news-detector' style='color:#58a6ff;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
