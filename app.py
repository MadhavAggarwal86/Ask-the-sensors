"""Web front end: upload a recording, ask a question, get the six-field grounded answer.

    streamlit run app.py

Same pipeline as scripts/answer.py: the small language model (Qwen3-1.7B, 4-bit, llama.cpp) reads the
question, the Python pipeline computes every number from the sensor timeline, and the model words
the explanation from those computed values.
"""
import hashlib
import io
import time

import altair as alt
import pandas as pd
import streamlit as st

from ats import config as C
from ats.io import load_csv
from ats.pipeline import SensorQA
from ats.qa.format import to_dict, to_text
from ats.qa.llm import SLM, SLM_MODELS, LLMExplainer, LLMParser

DEMO_DIR = C.DATA_DIR / "demo"
EXAMPLES = ["What was the user doing in the first 10 minutes?",
            "How long was the user walking?",
            "How many times did the user sit down?",
            "When did the user start walking?",
            "Did the user spend more time sitting or standing?",
            "Did the user sit for a long time without a break?",
            "Was the user exercising?",
            "What was the user's heart rate?"]

st.set_page_config(page_title="Ask the Sensors", layout="wide")


MODEL_NOTES = {"Qwen3-1.7B": "4-bit, 1.0 GB · more accurate, ~2–3 s per question",
               "Qwen3-0.6B": "4-bit, 378 MB · faster, ~1 s per question, less accurate"}


@st.cache_resource(show_spinner="Loading the activity model…")
def load_system() -> SensorQA:
    return SensorQA.from_path(C.MODELS_DIR / "hgb_full.joblib")


@st.cache_resource(show_spinner="Loading the language model…", max_entries=2)
def load_slm(name: str) -> tuple[LLMParser, LLMExplainer]:
    slm = SLM(SLM_MODELS[name])
    return LLMParser(slm), LLMExplainer(slm)


@st.cache_data(show_spinner=False, max_entries=8)
def read_recording(data: bytes):
    return load_csv(io.BytesIO(data))


def load_timeline(qa: SensorQA, name: str, data: bytes):
    """Preprocess + recognise + aggregate once per file; later questions reuse the timeline."""
    key = hashlib.sha1(data).hexdigest()
    if st.session_state.get("loaded") != key:
        t = time.perf_counter()
        qa.load(read_recording(data))
        st.session_state.update(loaded=key, load_s=time.perf_counter() - t, history=[])
    return qa.timeline


def timeline_chart(tl) -> alt.Chart:
    df = pd.DataFrame([{"activity": C.DISPLAY[iv.activity], "start (s)": iv.start, "end (s)": iv.end}
                       for iv in tl.intervals])
    return alt.Chart(df).mark_bar(height=18).encode(
        x=alt.X("start (s):Q", title=f"time ({C.TIME_BASE})"), x2="end (s):Q",
        y=alt.Y("activity:N", sort=[C.DISPLAY[c] for c in C.CLASSES], title=None),
        color=alt.Color("activity:N", legend=None),
        tooltip=["activity", "start (s)", "end (s)"]).properties(height=200)


st.title("Ask the Sensors")
st.caption("Grounded activity question answering from accelerometer and gyroscope recordings. "
           f"Timestamps are in {C.TIME_BASE}.")

qa = load_system()

with st.sidebar:
    st.header("Language model")
    available = [n for n, p in SLM_MODELS.items() if p.exists()]
    if not available:
        st.error("No language model found in models/slm/ (see README, Setup).")
        st.stop()
    model_name = st.radio("Reads the question and writes the explanation", available,
                          captions=[MODEL_NOTES.get(n, "") for n in available])
    qa.parser, qa.explainer = load_slm(model_name)

    st.header("1. Recording")
    uploads = st.file_uploader("Upload CSV recording(s)", type=["csv"], accept_multiple_files=True,
                               help="Columns: a time column (t / timestamp, seconds or ms) and acc_x, acc_y, "
                                    "acc_z; gyro_x, gyro_y, gyro_z are optional.")
    files = {u.name: u.getvalue() for u in uploads or []}
    if not files:
        files = {p.name: p.read_bytes() for p in sorted(DEMO_DIR.glob("*.csv"))}
        if files:
            st.info("No file uploaded: using the demo recording.")
    if not files:
        st.stop()
    name = st.selectbox("Recording", list(files))

try:
    tl = load_timeline(qa, name, files[name])
except Exception as e:     # a malformed upload should explain itself, not crash the page
    st.error(f"Could not read {name}: {e}")
    st.stop()

with st.sidebar:
    st.markdown(f"**Length:** {qa.duration / 60:.1f} min  \n"
                f"**Sensors:** {'accelerometer + gyroscope' if tl.has_gyro else 'accelerometer only'}")
    st.caption(f"Processed in {st.session_state['load_s']:.1f} s · {len(tl.intervals)} activity intervals")

with st.expander("Recognised activity timeline", expanded=False):
    if tl.intervals:
        st.altair_chart(timeline_chart(tl), use_container_width=True)
    else:
        st.write("No usable windows in this recording.")

st.header("2. Ask a question")
example = st.selectbox("Examples", ["(type your own)"] + EXAMPLES)
with st.form("ask"):
    question = st.text_input("Question", value="" if example.startswith("(") else example)
    asked = st.form_submit_button("Ask", type="primary")

if asked and question.strip():
    t = time.perf_counter()
    with st.spinner("Reading the question, computing the answer, writing the explanation…"):
        ans = qa.ask(question.strip())
    st.session_state["history"].insert(0, (question.strip(), ans, time.perf_counter() - t, model_name))

for i, (q, ans, secs, used) in enumerate(st.session_state.get("history", [])):
    st.subheader(("Answer" if i == 0 else "Earlier") + f" · {used} · {secs:.1f} s")
    st.code(to_text(q, ans), language=None, wrap_lines=True)
    with st.expander("JSON"):
        st.json(to_dict(q, ans))
