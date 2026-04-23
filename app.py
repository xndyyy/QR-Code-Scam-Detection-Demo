import os
import random
from PIL import Image
import streamlit as st
import pandas as pd
import xgboost as xgb

from utils.decode_qr import decode_qr_from_array
from utils.predict import (
    load_model,
    load_feature_columns,
    load_cnn_model,
    predict_url_model_probabilities,
    predict_cnn_probability,
    predict_stacked_probability,
)
from utils.explain import (
    build_feature_row_from_dict,
    get_xgb_local_contributions,
    get_contribution_strength,
    describe_xgb_contribution,
    FEATURE_LABELS
)

st.set_page_config(
    page_title="QR Code Scam Detection Demo",
    layout="wide",
)

st.markdown("""
<style>
html, body, [class*="css"]  {
    font-size: 22px !important;
}

h1 { font-size: 32px !important; }
h2 { font-size: 28px !important; }
h3 { font-size: 26px !important; }

section[data-testid="stSidebar"] * {
    font-size: 18px !important;
}

button {
    font-size: 18px !important;
}

div[data-testid="stMetricValue"] {
    font-size: 28px !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 18px !important;
}

code {
    font-size: 18px !important;
}
            
.explain-card {
    border: 1px solid #e6e6e6;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
    background-color: #ffffff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

.explain-rank {
    font-size: 21px;
    font-weight: 700;
    color: #666666;
    margin-bottom: 4px;
}

.explain-title {
    font-size: 24px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 8px;
}

.explain-meta {
    font-size: 21px;
    margin-bottom: 8px;
    line-height: 1.5;
}

.explain-value {
    font-weight: 700;
    color: #111827;
}

.explain-contrib-pos {
    font-weight: 700;
    color: #dc2626;
}

.explain-contrib-neg {
    font-weight: 700;
    color: #15803d;
}

.explain-body {
    font-size: 21px;
    color: #374151;
    line-height: 1.6;
    margin-top: 6px;
}

.explain-header-box {
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 16px;
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
}
            
</style>
""", unsafe_allow_html=True)

st.title("QR Code Scam Detection Demo")
st.write(
    "This demo classifies a QR code as **Benign** or **Phishing** using "
    "four URL-based classifiers, one CNN-based image classifier, and a "
    "stacking ensemble."
)

@st.cache_resource
def load_all_models():
    models = {
        "lr": load_model("models/lr_tuned.joblib"),
        "rf": load_model("models/rf_tuned.joblib"),
        "xgb": load_model("models/xgb_tuned.joblib"),
        "mlp": load_model("models/mlp_tuned.joblib"),
    }
    stacker = load_model("models/stacked_model.pkl")
    feature_columns = load_feature_columns("models/feature_columns.pkl")
    cnn_model = load_cnn_model("models/cnn_final.pth", device="cpu")
    return models, stacker, feature_columns, cnn_model


models, stacker, feature_columns, cnn_model = load_all_models()

# -----------------------------
# Session state
# -----------------------------
if "decoded_url" not in st.session_state:
    st.session_state.decoded_url = None

if "feature_dict" not in st.session_state:
    st.session_state.feature_dict = None

if "sample_image_path" not in st.session_state:
    st.session_state.sample_image_path = None

if "sample_filename" not in st.session_state:
    st.session_state.sample_filename = None

if "last_selected_sample" not in st.session_state:
    st.session_state.last_selected_sample = None

if "last_image_key" not in st.session_state:
    st.session_state.last_image_key = None

# -----------------------------
# Helpers
# -----------------------------
def get_image_files(folder_path: str):
    if not os.path.isdir(folder_path):
        return []
    return [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def choose_random_sample(category: str):
    folder_path = os.path.join("samples", category)
    files = get_image_files(folder_path)

    if not files:
        return None, None

    chosen_file = random.choice(files)
    return os.path.join(folder_path, chosen_file), chosen_file


# -----------------------------
# Sample config
# -----------------------------
SAMPLE_INFO = {
    "None": None,
    "Benign Sample": {
        "category": "benign",
        "true_label": "Benign",
    },
    "Phishing Sample": {
        "category": "phishing",
        "true_label": "Phishing",
    },
}

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Input")

selected_sample = st.sidebar.selectbox(
    "Choose a QR sample",
    list(SAMPLE_INFO.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Or upload a QR code image",
    type=["png", "jpg", "jpeg"]
)

refresh_sample = st.sidebar.button("Refresh Sample")
show_feature_details = st.sidebar.checkbox("Show extracted URL features", value=True)
show_model_breakdown = st.sidebar.checkbox("Show individual model probabilities", value=True)

decode_clicked = st.sidebar.button("Decode QR", type="primary")
predict_clicked = st.sidebar.button("Predict", type="primary")

# -----------------------------
# Resolve input image
# -----------------------------
image = None
input_source = None
true_label = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    input_source = "Uploaded Image"
    true_label = None
    current_image_key = f"upload::{uploaded_file.name}"

else:
    sample_config = SAMPLE_INFO[selected_sample]
    current_image_key = None

    if sample_config is not None:
        needs_new_sample = (
            st.session_state.last_selected_sample != selected_sample
            or refresh_sample
            or st.session_state.sample_image_path is None
        )

        if needs_new_sample:
            sample_path, sample_filename = choose_random_sample(sample_config["category"])
            st.session_state.sample_image_path = sample_path
            st.session_state.sample_filename = sample_filename
            st.session_state.last_selected_sample = selected_sample

        if st.session_state.sample_image_path is not None:
            image = Image.open(st.session_state.sample_image_path)
            input_source = f"{selected_sample}: {st.session_state.sample_filename}"
            true_label = sample_config["true_label"]
            current_image_key = f"sample::{st.session_state.sample_image_path}"

# -----------------------------
# Guard: no image
# -----------------------------
if image is None:
    st.info("Select a sample from the sidebar or upload a QR code image.")
    st.stop()

# -----------------------------
# Reset decode/predict state when image changes
# -----------------------------
if st.session_state.last_image_key != current_image_key:
    st.session_state.decoded_url = None
    st.session_state.feature_dict = None
    st.session_state.last_image_key = current_image_key

# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1, 1.8])

with left_col:
    st.header("QR Code Input")
    st.image(image, width=480)
    st.caption(input_source)

    if true_label is not None:
        st.markdown(f"**Ground Truth:** {true_label}")

with right_col:
    st.header("Pipeline Output")

    # -------- Step 1: Decode --------
    if decode_clicked:
        with st.spinner("Decoding QR code..."):
            decoded_url = decode_qr_from_array(image)

            if not decoded_url:
                st.error("Could not decode QR code from the input image.")
                st.session_state.decoded_url = None
                st.session_state.feature_dict = None
                st.stop()

            st.session_state.decoded_url = decoded_url

            _, feature_dict = predict_url_model_probabilities(
                decoded_url,
                models,
                feature_columns
            )
            st.session_state.feature_dict = feature_dict

    if st.session_state.decoded_url:
        st.markdown("## Step 1: Decoded URL")
        st.code(st.session_state.decoded_url)

        if show_feature_details and st.session_state.feature_dict is not None:
            feature_dict = st.session_state.feature_dict

            st.markdown("## URL Feature Snapshot")

            f1, f2, f3, f4 = st.columns(4)
            with f1:
                st.metric("URL Length", feature_dict["url_length"])
            with f2:
                st.metric("Domain Length", feature_dict["domain_length"])
            with f3:
                st.metric("Subdomains", feature_dict["num_subdomains"])
            with f4:
                st.metric("Path Length", feature_dict["path_length"])

            f5, f6, f7, f8 = st.columns(4)
            with f5:
                st.metric("Number of Dots", feature_dict["num_dot"])
            with f6:
                st.metric("Special Characters", feature_dict["num_special"])
            with f7:
                st.metric("Number of Tokens", feature_dict["num_tokens"])
            with f8:
                st.metric("Longest Token Length", feature_dict["longest_token_length"])

    else:
        st.info("Click **Decode QR** to extract the URL.")
        st.stop()

    # -------- Step 2: Predict --------
    if not predict_clicked:
        st.info("Click **Predict** to run the classification models.")
        st.stop()

    try:
        with st.spinner("Running multimodal prediction pipeline..."):
            url_probs, _ = predict_url_model_probabilities(
                st.session_state.decoded_url,
                models,
                feature_columns
            )
            cnn_prob = predict_cnn_probability(image, cnn_model, device="cpu")
            final_prob = predict_stacked_probability(url_probs, cnn_prob, stacker)
            predicted_label = "Phishing" if final_prob >= 0.5 else "Benign"

        st.markdown("## Step 2: Prediction Result")

        col1, col2 = st.columns(2)
        color = "red" if predicted_label == "Phishing" else "green"

        with col1:
            st.markdown(
                f"<h1 style='color:{color}; margin-top:0; margin-bottom:0;'>{predicted_label}</h1>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"<h1 style='margin-top:0; margin-bottom:0;'>Probability: {final_prob:.4f}</h1>",
                unsafe_allow_html=True
            )

        if true_label is not None:
            if predicted_label == true_label:
                st.success(f"Prediction matches ground truth: {true_label}")
            else:
                st.warning(
                    f"Prediction differs from ground truth. "
                    f"Ground truth: {true_label}, Predicted: {predicted_label}"
                )

        if show_model_breakdown:
            st.markdown("## Model Probability Breakdown")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Logistic Regression", f"{url_probs['lr']:.4f}")
                st.metric("XGBoost", f"{url_probs['xgb']:.4f}")
                st.metric("Custom CNN", f"{cnn_prob:.4f}")

            with c2:
                st.metric("Random Forest", f"{url_probs['rf']:.4f}")
                st.metric("MLP", f"{url_probs['mlp']:.4f}")
        
        st.markdown("## Key Feature Contributions")

        st.caption(
            "This section explains the decoded URL using local feature contributions from the XGBoost URL model."
            " Positive contributions push the score toward phishing, while negative contributions push it toward benign."
            " The final classification shown above still comes from the stacking ensemble."
        )

        if st.session_state.feature_dict is not None:
            x_row = build_feature_row_from_dict(st.session_state.feature_dict, feature_columns)
            contrib_df, bias = get_xgb_local_contributions(models["xgb"], x_row, top_n=5)

            total_abs = contrib_df["abs_contribution"].sum()

            for rank, (_, row) in enumerate(contrib_df.iterrows(), start=1):
                feature = row["feature"]
                pretty_name = FEATURE_LABELS.get(feature, feature)
                value = row["value"]
                contribution = row["contribution"]

                direction = "Phishing" if contribution > 0 else "Benign"
                contrib_class = "explain-contrib-pos" if contribution > 0 else "explain-contrib-neg"
                strength = get_contribution_strength(contribution)

                # nicer formatting for floats
                if isinstance(value, float):
                    if value.is_integer():
                        display_value = int(value)
                    else:
                        display_value = round(value, 3)
                else:
                    display_value = value

                explanation_text = describe_xgb_contribution(feature, value, contribution)

                st.markdown(
                    f"""
                    <div class='explain-card'>
                        <div class='explain-rank'>Rank #{rank}</div>
                        <div class='explain-title'>{pretty_name}</div>
                        <div class='explain-meta'>
                            <span class='explain-value'>Value:</span> {display_value}<br>
                            <span class='explain-value'>Contribution:</span>
                            <span class='{contrib_class}'>{contribution:+.4f}</span>
                            ({strength} push toward {direction})<br>
                        </div>
                        <div class='explain-body'>
                            {explanation_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

st.markdown("---")
st.caption(
    "Demo note: this prototype is designed for presentation purposes to illustrate "
    "the end-to-end multimodal QR scam detection pipeline."
)