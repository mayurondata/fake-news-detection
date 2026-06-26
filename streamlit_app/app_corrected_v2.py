"""
app.py  ─  Fake News Detector  ·  Streamlit UI
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predictor import get_predictor, preprocess

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
)

if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""
if "selected_example" not in st.session_state:
    st.session_state["selected_example"] = "Choose an example"
if "history" not in st.session_state:
    st.session_state["history"] = []
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

EXAMPLES = {
    "Choose an example": "",
    "Health breakthrough": (
        "A report says that researchers at a well-known university developed a new "
        "treatment that reduced symptoms in patients during an early clinical study. "
        "The article quotes doctors involved in the research, mentions the sample size, "
        "and includes comments about the need for larger trials before broad conclusions are made."
    ),
    "Election claim": (
        "A viral social media post claims that election officials secretly destroyed "
        "thousands of ballots in a major city and that several unnamed insiders have "
        "confirmed the cover-up. The post does not provide official documents, named "
        "sources, or verifiable evidence, but it urges people to share the claim immediately."
    ),
    "Celebrity rumor": (
        "Several entertainment blogs are claiming that a famous actor was arrested at a "
        "private event last night after a confrontation with security staff. The stories "
        "mostly repeat each other, cite anonymous attendees, and provide no police statement "
        "or direct confirmation from the actor's representatives."
    ),
}


def apply_example():
    st.session_state["article_text"] = EXAMPLES.get(st.session_state["selected_example"], "")


def clear_inputs():
    st.session_state["article_text"] = ""
    st.session_state["selected_example"] = "Choose an example"
    st.session_state["run_analysis"] = False


def trigger_analysis():
    st.session_state["run_analysis"] = True


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap');
:root {
    --bg:#0d1117; --surface:#161b22; --border:#30363d; --green:#2ecc71;
    --red:#e74c3c; --blue:#58a6ff; --yellow:#f0c14b; --muted:#8b949e; --text:#e6edf3;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: var(--bg) !important; }
[data-testid="stSidebar"], section[data-testid="stSidebar"], div[data-testid="collapsedControl"] { display:none !important; }
p, li, label, .stMarkdown { color: var(--text) !important; font-family: 'Inter', sans-serif; }
.hero-title { font-family:'Playfair Display', serif; font-size:clamp(2rem,5vw,3.6rem); font-weight:900; letter-spacing:-1px; line-height:1.1; color:var(--text); margin:0; }
.hero-sub { font-family:'Inter', sans-serif; font-size:1rem; color:var(--muted); margin-top:.5rem; }
.hero-badge { display:inline-block; background:linear-gradient(135deg, #1f6feb33, #58a6ff22); border:1px solid #1f6feb; color:var(--blue); font-size:.7rem; font-family:'JetBrains Mono', monospace; padding:3px 10px; border-radius:20px; margin-bottom:.8rem; letter-spacing:.08em; }
.verdict-real { background:#0d2818; border:2px solid var(--green); border-radius:16px; padding:2rem; text-align:center; }
.verdict-fake { background:#2d0d0d; border:2px solid var(--red); border-radius:16px; padding:2rem; text-align:center; }
.verdict-label { font-family:'Playfair Display', serif; font-size:3rem; font-weight:900; letter-spacing:4px; }
.verdict-conf { font-family:'Inter', sans-serif; font-size:.9rem; color:var(--muted); margin-top:.3rem; }
.metric-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.2rem 1.5rem; text-align:center; }
.metric-value { font-family:'Playfair Display', serif; font-size:2.2rem; font-weight:700; color:var(--blue); }
.metric-label { font-family:'Inter', sans-serif; font-size:.8rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
textarea { background:var(--surface) !important; color:var(--text) !important; border:1px solid var(--border) !important; border-radius:10px !important; font-family:'Inter', sans-serif !important; font-size:.95rem !important; }
textarea:focus { border-color:var(--blue) !important; }
.stButton > button { background:linear-gradient(135deg, #1f6feb, #388bfd) !important; color:white !important; border:none !important; border-radius:10px !important; font-family:'Inter', sans-serif !important; font-weight:600 !important; padding:.7rem 2.5rem !important; font-size:1rem !important; transition:opacity .2s !important; width:100%; }
.stButton > button:hover { opacity:.85 !important; }
.history-row { display:flex; align-items:center; gap:.8rem; padding:.7rem 1rem; border-bottom:1px solid var(--border); font-family:'Inter', sans-serif; font-size:.85rem; color:var(--text); }
.pill-real { background:#0d2818; color:var(--green); border:1px solid var(--green); border-radius:20px; padding:2px 10px; font-size:.75rem; }
.pill-fake { background:#2d0d0d; color:var(--red); border:1px solid var(--red); border-radius:20px; padding:2px 10px; font-size:.75rem; }
hr { border-color:var(--border) !important; margin:2rem 0 !important; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-badge">AI-ASSISTED VERIFICATION</div>
<h1 class="hero-title">Fake News Detector</h1>
<p class="hero-sub">Paste an article, try a sample, and get a quick classification with confidence.</p>
""",
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)

col_input, col_result = st.columns([1.15, 1], gap="large")

with col_input:
    st.markdown("#### 📰 Article Text")

    st.selectbox(
        "Try an example",
        options=list(EXAMPLES.keys()),
        key="selected_example",
        on_change=apply_example,
    )

    article_text = st.text_area(
        "Paste news article text",
        key="article_text",
        height=280,
        placeholder="Paste a news headline + body text here...\n\nTip: more text = more accurate prediction.",
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        st.button("🔍 Analyse Article", use_container_width=True, on_click=trigger_analysis)
    with col_clear:
        st.button("✕ Clear", use_container_width=True, on_click=clear_inputs)

    if article_text.strip():
        clean = preprocess(article_text)
        n_words = len(article_text.split())
        n_tokens = len(clean.split())
        st.markdown(
            f"<div style='color:#8b949e; font-size:.8rem; font-family:Inter,sans-serif; margin-top:.5rem;'>📝 {n_words} words → <span style='color:#58a6ff;'>{n_tokens} tokens</span> after preprocessing</div>",
            unsafe_allow_html=True,
        )

with col_result:
    st.markdown("#### 🎯 Verdict")

    if st.session_state["run_analysis"]:
        if not st.session_state["article_text"].strip():
            st.warning("⚠️ Please enter some article text first.")
            st.session_state["run_analysis"] = False
        elif len(st.session_state["article_text"].split()) < 10:
            st.warning("⚠️ Article too short — add at least 10 words for a reliable prediction.")
            st.session_state["run_analysis"] = False
        else:
            with st.spinner("Loading model & analysing..."):
                predictor = get_predictor()
                t0 = time.time()
                result = predictor.predict(st.session_state["article_text"])
                elapsed = time.time() - t0

            icon = "✅" if result.label == "REAL" else "🚨"
            cls = "verdict-real" if result.label == "REAL" else "verdict-fake"
            color = "#2ecc71" if result.label == "REAL" else "#e74c3c"
            conf_pct = f"{result.confidence * 100:.1f}%"

            st.markdown(
                f"""
                <div class='{cls}'>
                    <div class='verdict-label' style='color:{color};'>{icon} {result.label}</div>
                    <div class='verdict-conf'>Confidence: <b style='color:{color};'>{conf_pct}</b></div>
                    <div class='verdict-conf' style='margin-top:.2rem;'>Inference: {elapsed * 1000:.0f}ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prob_df = pd.DataFrame(
                {
                    "Label": ["REAL", "FAKE"],
                    "Confidence": [
                        result.confidence if result.label == "REAL" else 1 - result.confidence,
                        result.confidence if result.label == "FAKE" else 1 - result.confidence,
                    ],
                }
            )

            fig_bar = px.bar(
                prob_df,
                x="Label",
                y="Confidence",
                color="Label",
                color_discrete_map={"REAL": "#2ecc71", "FAKE": "#e74c3c"},
                template="plotly_dark",
            )
            fig_bar.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
                showlegend=False,
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result.confidence * 100,
                    number={"suffix": "%", "font": {"color": "#e6edf3"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                        "bar": {"color": color},
                        "bgcolor": "#161b22",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 50], "color": "#2d0d0d"},
                            {"range": [50, 100], "color": "#0d2818"},
                        ],
                    },
                    title={"text": "Model Confidence", "font": {"color": "#e6edf3"}},
                )
            )
            fig_gauge.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.session_state["history"].insert(
                0,
                {
                    "label": result.label,
                    "confidence": conf_pct,
                    "elapsed_ms": f"{elapsed * 1000:.0f}",
                    "preview": st.session_state["article_text"][:90].replace("\n", " ") + ("..." if len(st.session_state["article_text"]) > 90 else ""),
                },
            )
            st.session_state["run_analysis"] = False
    else:
        st.info("Run an analysis to see the prediction, confidence, and charts.")

st.markdown("<hr>", unsafe_allow_html=True)

metric_1, metric_2, metric_3 = st.columns(3)
with metric_1:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(st.session_state['history'])}</div><div class='metric-label'>Analyses Run</div></div>", unsafe_allow_html=True)
with metric_2:
    real_count = sum(1 for item in st.session_state['history'] if item['label'] == 'REAL')
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{real_count}</div><div class='metric-label'>Real Predictions</div></div>", unsafe_allow_html=True)
with metric_3:
    fake_count = sum(1 for item in st.session_state['history'] if item['label'] == 'FAKE')
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{fake_count}</div><div class='metric-label'>Fake Predictions</div></div>", unsafe_allow_html=True)

st.markdown("### Recent Checks")
if st.session_state["history"]:
    for item in st.session_state["history"][:8]:
        pill_class = "pill-real" if item["label"] == "REAL" else "pill-fake"
        st.markdown(
            f"<div class='history-row'><span class='{pill_class}'>{item['label']}</span><span>{item['confidence']}</span><span>{item['elapsed_ms']} ms</span><span style='color:#8b949e'>{item['preview']}</span></div>",
            unsafe_allow_html=True,
        )
else:
    st.caption("No analyses yet.")
