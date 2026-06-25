"""
Streamlit demo: upload or sample a synthetic image and get a caption from
the trained MultiHead CNN or CNN-LSTM models.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.multihead_inference import load_model as load_multihead, predict as predict_multihead
from src.inference.lstm_inference import build_vocab, load_model as load_lstm, predict as predict_lstm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Sentence accuracy on test set (from Evaluation/metrics_*.json) ──────────
PERF = {
    "tictactoe": {"MultiHead CNN": "96.7%", "CNN-LSTM": "—"},
    "shapes":    {"MultiHead CNN": "25.0%", "CNN-LSTM": "0%"},
    "numbers":   {"MultiHead CNN": "4.0%",  "CNN-LSTM": "—"},
}

# ── Cached model loaders ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_multihead(task: str):
    return load_multihead(task, PROJECT_ROOT, DEVICE)


@st.cache_resource(show_spinner=False)
def get_lstm(task: str):
    vocab = build_vocab(task, PROJECT_ROOT)
    model = load_lstm(task, PROJECT_ROOT, len(vocab.word2idx), DEVICE)
    return model, vocab


# ── Dataset helpers ──────────────────────────────────────────────────────────

def load_test_rows(task: str) -> list[dict]:
    metadata_dir = PROJECT_ROOT / "data" / "processed" / task / "metadata"
    single = metadata_dir / f"{task}_metadata.jsonl"
    if single.exists():
        with open(single, "r", encoding="utf-8") as f:
            return [r for line in f if line.strip() for r in [json.loads(line)] if r.get("split") == "test"]
    test_file = metadata_dir / "test.jsonl"
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return []


def image_path(task: str, row: dict) -> Path:
    return PROJECT_ROOT / "data" / "processed" / task / row["image_path"]


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    task = st.selectbox(
        "Dataset / Task",
        ["tictactoe", "shapes", "numbers"],
        format_func=lambda x: x.title(),
    )

    model_choice = st.selectbox("Model", ["MultiHead CNN", "CNN-LSTM"])

    st.divider()

    st.subheader("Test accuracy")
    perf = PERF[task]
    col_a, col_b = st.columns(2)
    col_a.metric("MultiHead CNN", perf["MultiHead CNN"])
    col_b.metric("CNN-LSTM", perf["CNN-LSTM"])
    st.caption("Sentence accuracy on held-out test set")

    st.divider()

    st.info(
        "Models are trained on **synthetically generated images** (PIL). "
        "Uploading a real photograph will produce unpredictable output. "
        "Use **Random test sample** for reliable results."
    )

    with st.expander("GPT-3.5 comparison"):
        st.markdown(
            """
| Task | MultiHead CNN | GPT-3.5 |
|---|---|---|
| TicTacToe | **96.7%** | 82% |
| Shapes | 25% | 48% |
| Numbers | 4% | 72% |
"""
        )
        st.caption("MultiHead CNN beats GPT-3.5 on TicTacToe.")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🖼️ Image Caption Generator")
st.caption(
    "Group 40 · ITAI Coursework — MultiHead CNN vs CNN-LSTM on synthetic image datasets"
)

col_left, col_right = st.columns([1, 1], gap="large")

# ── Left column: image input ──────────────────────────────────────────────────

with col_left:
    st.subheader("Input Image")

    source = st.radio(
        "Image source",
        ["Random test sample", "Upload image"],
        horizontal=True,
    )

    selected_image: Image.Image | None = None
    true_caption: str | None = None

    if source == "Upload image":
        uploaded = st.file_uploader(
            "Choose a PNG or JPG",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded:
            selected_image = Image.open(uploaded).convert("RGB")
            st.image(selected_image, use_container_width=True)

    else:
        rows = load_test_rows(task)

        if not rows:
            st.warning(
                f"No test data found for **{task}**. "
                "Run the data generation script first, or switch to **Upload image**."
            )
        else:
            # Initialise or refresh sample when task changes
            if (
                st.button("🔀 New random sample")
                or "sample_row" not in st.session_state
                or st.session_state.get("sample_task") != task
            ):
                st.session_state["sample_row"] = random.choice(rows)
                st.session_state["sample_task"] = task

            row = st.session_state["sample_row"]
            img_path = image_path(task, row)

            if img_path.exists():
                selected_image = Image.open(img_path).convert("RGB")
                true_caption = row["caption"]
                st.image(selected_image, use_container_width=True)
                st.caption(f"ID: `{row['id']}`")
            else:
                st.error(f"Image file not found: `{img_path}`")

# ── Right column: prediction ──────────────────────────────────────────────────

with col_right:
    st.subheader("Prediction")

    if selected_image is None:
        st.info("← Load a test sample or upload an image to run inference.")
    else:
        with st.spinner(f"Running {model_choice}…"):
            try:
                if model_choice == "MultiHead CNN":
                    model, label_maps = get_multihead(task)
                    caption, head_results = predict_multihead(model, label_maps, selected_image, task, DEVICE)

                    st.markdown(f"### {caption}")

                    if true_caption is not None:
                        match = caption == true_caption
                        st.markdown(
                            f"**True caption:** {true_caption}  \n"
                            f"{'✅ Correct' if match else '❌ Wrong'}"
                        )

                    st.divider()
                    st.markdown("**Per-attribute confidence**")

                    for head, result in head_results.items():
                        label = result["label"]
                        conf = result["confidence"]
                        # Sort probabilities descending for the bar chart
                        probs_sorted = sorted(result["probs"].items(), key=lambda x: -x[1])

                        head_col, label_col, bar_col = st.columns([2, 1, 3])
                        head_col.write(f"`{head}`")
                        label_col.write(f"**{label}**")
                        bar_col.progress(conf, text=f"{conf:.0%}")

                        with st.expander(f"All options for '{head}'", expanded=False):
                            for lbl, prob in probs_sorted:
                                st.write(f"{lbl}: {prob:.1%}")

                else:  # CNN-LSTM
                    model, vocab = get_lstm(task)
                    caption = predict_lstm(model, vocab, selected_image, DEVICE)

                    st.markdown(f"### {caption}")

                    if true_caption is not None:
                        match = caption == true_caption
                        st.markdown(
                            f"**True caption:** {true_caption}  \n"
                            f"{'✅ Correct' if match else '❌ Wrong'}"
                        )

                    if task in ("shapes", "numbers"):
                        st.warning(
                            "CNN-LSTM achieves 0% sentence accuracy on shapes "
                            "and low accuracy on numbers — output may be incoherent."
                        )

            except FileNotFoundError as exc:
                st.error(f"**File not found:** {exc}")
                st.caption("Make sure model weights exist in `artifacts/` and data exists in `data/processed/`.")
            except Exception as exc:
                st.error(f"Inference error: {exc}")
                st.exception(exc)
