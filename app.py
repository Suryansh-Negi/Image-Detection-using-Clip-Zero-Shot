import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import pipeline


st.set_page_config(page_title="Image Detection App", page_icon="📷", layout="centered")

st.title("High Accuracy Image Prediction (Zero-Shot CLIP)")
st.write(
    "This app uses CLIP zero-shot classification. "
    "It compares the image against your target labels and returns the best matches."
)


@st.cache_resource  # Add this decorator to cache the model     
def load_model():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-large-patch14",
    )


classifier = load_model()

DEFAULT_LABELS = [
    "dog",
    "cat",
    "person",
    "mobile phone",
    "laptop",
    "keyboard",
    "mouse",
    "monitor",
    "tablet",
    "headphones",
    "smartwatch",
    "leaf",
    "flower",
    "plant",
    "tree",
    "fruit",
    "vegetable",
    "car",
    "motorcycle",
    "bicycle",
    "bus",
    "bottle",
    "water bottle",
    "book",
    "notebook",
    "pen",
    "cup",
    "mug",
    "chair",
    "table",
]


def parse_labels(raw_text: str):
    labels = [x.strip().lower() for x in raw_text.split(",")]
    labels = [x for x in labels if x]
    # Preserve order and remove duplicates
    return list(dict.fromkeys(labels))


def predict_from_pil(pil_img: Image.Image, candidate_labels, top_k: int = 5):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    preds = classifier(pil_img, candidate_labels=candidate_labels)
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    return preds[:top_k]


def render_predictions(predictions, confidence_threshold: float):
    st.subheader("Predictions")
    if not predictions:
        st.warning("No predictions returned by model.")
        return

    best = predictions[0]
    if best["score"] < confidence_threshold:
        st.error(
            f"Low confidence ({best['score'] * 100:.2f}%). "
            "Prediction marked as Unknown. Try clearer image or better labels."
        )
    else:
        st.success(
            f"Best Match: {best['label'].title()} ({best['score'] * 100:.2f}%)"
        )

    for pred in predictions:
        st.write(f"- **{pred['label'].title()}**: {pred['score'] * 100:.2f}%")

    st.markdown("#### Confidence distribution")
    chart_df = pd.DataFrame(
        [
            {"label": p["label"].title(), "confidence_pct": round(p["score"] * 100, 2)}
            for p in predictions
        ]
    )
    chart_df["value_label"] = chart_df["confidence_pct"].map(lambda v: f"{v:.1f}%")

    bar_height = 22
    chart_h = max(220, len(predictions) * (bar_height + 14) + 72)

    base = alt.Chart(chart_df).encode(
        y=alt.Y(
            "label:N",
            sort=alt.EncodingSortField(field="confidence_pct", order="descending"),
            title=None,
            axis=alt.Axis(
                labelFontSize=12,
                labelFontWeight=500,
                labelColor="#334155",
                labelLimit=280,
                ticks=False,
                domain=False,
                offset=6,
            ),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Class"),
            alt.Tooltip("confidence_pct:Q", title="Confidence", format=".2f"),
        ],
    )

    bars = base.mark_bar(
        size=bar_height,
        cornerRadiusEnd=5,
    ).encode(
        x=alt.X(
            "confidence_pct:Q",
            title=None,
            axis=alt.Axis(
                title="Match strength (%)",
                titleFontSize=12,
                titleFontWeight=500,
                titleColor="#64748b",
                titlePadding=12,
                labelFontSize=11,
                labelColor="#64748b",
                tickCount=6,
                ticks=True,
                tickSize=4,
                domainColor="#cbd5e1",
                grid=True,
                gridColor="#e2e8f0",
                gridDash=[2, 4],
                format="d",
            ),
            scale=alt.Scale(domain=[0, 100], nice=False, paddingOuter=0.02),
        ),
        color=alt.Color(
            "confidence_pct:Q",
            scale=alt.Scale(
                scheme="blues",
                domainMin=0,
                domainMax=100,
            ),
            legend=None,
        ),
    )

    value_labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=8,
        fontSize=12,
        fontWeight=600,
        color="#0f172a",
    ).encode(
        x=alt.X("confidence_pct:Q"),
        text="value_label:N",
    )

    chart = (
        (bars + value_labels)
        .properties(
            height=chart_h,
            padding={"left": 4, "right": 36, "top": 4, "bottom": 8},
            title=alt.TitleParams(
                text="Model scores by candidate label",
                subtitle="CLIP zero-shot: higher values indicate stronger image–text alignment",
                fontSize=15,
                fontWeight=600,
                color="#0f172a",
                subtitleFontSize=11,
                subtitleFontWeight=400,
                subtitleColor="#64748b",
                anchor="start",
                offset=10,
            ),
        )
        .configure(
            background="transparent",
            padding={"top": 8},
            font="system-ui, -apple-system, 'Segoe UI', sans-serif",
        )
        .configure_view(
            strokeWidth=0,
            fill="#f8fafc",
            cornerRadius=8,
        )
    )
    st.altair_chart(chart, use_container_width=True)


st.subheader("Label Setup (Very Important for Accuracy)")
labels_text = st.text_area(
    "Enter possible labels separated by commas",
    value=", ".join(DEFAULT_LABELS),
    help="Accuracy improves when labels are relevant to what you want to detect.",
)

candidate_labels = parse_labels(labels_text)
if len(candidate_labels) < 2:
    st.warning("Please enter at least 2 labels.")
    st.stop()

confidence_threshold = st.slider(
    "Unknown threshold (minimum confidence)",
    min_value=0.10,
    max_value=0.95,
    value=0.45,
    step=0.01,
    help="If best score is below this value, app returns Unknown.",
)


mode = st.radio("Choose input method:", ["Upload Image", "Webcam Capture"], horizontal=True)

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict (Upload)"):
            with st.spinner("Running prediction..."):
                predictions = predict_from_pil(pil_img, candidate_labels)
            render_predictions(predictions, confidence_threshold)

else:
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        pil_img = Image.open(camera_input)
        st.image(pil_img, caption="Captured Image", use_container_width=True)

        if st.button("Predict (Webcam)"):
            with st.spinner("Running prediction..."):
                predictions = predict_from_pil(pil_img, candidate_labels)
            render_predictions(predictions, confidence_threshold)

st.markdown("---")
st.caption(
    "Tip: For high accuracy, keep labels limited to your use-case "
    "(example: dog, cat, mobile phone, leaf, bottle, laptop)."
)

