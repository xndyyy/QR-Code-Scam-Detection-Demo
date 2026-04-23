import joblib
import numpy as np
from PIL import Image

from utils.decode_qr import decode_qr_from_array
from utils.predict import (
    load_cnn_model,
    predict_url_model_probabilities,
    predict_cnn_probability,
    predict_stacked_probability,
)


def main():
    print("Loading models...")

    models = {
        "lr": joblib.load("models/lr_tuned.joblib"),
        "rf": joblib.load("models/rf_tuned.joblib"),
        "xgb": joblib.load("models/xgb_tuned.joblib"),
        "mlp": joblib.load("models/mlp_tuned.joblib"),
    }
    stacker = joblib.load("models/stacked_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    cnn_model = load_cnn_model("models/cnn_final.pth", device="cpu")

    print("All models loaded successfully.")

    sample_path = "samples/qr_1.png"
    print(f"\nOpening sample image: {sample_path}")
    image = Image.open(sample_path)

    print("Decoding QR...")
    decoded_url = decode_qr_from_array(image)
    print("Decoded URL:", decoded_url)

    if not decoded_url:
        raise ValueError("QR decoding failed. No URL returned.")

    print("\nRunning URL-based models...")
    url_probs, feature_dict = predict_url_model_probabilities(
        decoded_url, models, feature_columns
    )

    print("LR probability: ", url_probs["lr"])
    print("RF probability: ", url_probs["rf"])
    print("XGB probability:", url_probs["xgb"])
    print("MLP probability:", url_probs["mlp"])

    print("\nRunning CNN...")
    cnn_prob = predict_cnn_probability(image, cnn_model, device="cpu")
    print("CNN probability:", cnn_prob)

    print("\nRunning stacked model...")
    final_prob = predict_stacked_probability(url_probs, cnn_prob, stacker)
    final_label = "Phishing" if final_prob >= 0.5 else "Benign"

    print("Final stacked probability:", final_prob)
    print("Final prediction:", final_label)

    print("\nFeature count:", len(feature_dict))
    print("Expected feature count:", len(feature_columns))

    missing = [col for col in feature_columns if col not in feature_dict]
    extra = [col for col in feature_dict if col not in feature_columns]

    print("Missing features:", missing)
    print("Extra features:", extra)

    if missing:
        raise ValueError(f"Missing expected features: {missing}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()