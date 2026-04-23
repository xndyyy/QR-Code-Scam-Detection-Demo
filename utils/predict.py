import joblib
import numpy as np
import torch
from PIL import Image

from utils.cnn_model import CNN
from utils.feature_engineering import extract_url_features


def load_model(path: str):
    return joblib.load(path)


def load_feature_columns(path: str):
    return joblib.load(path)


def load_cnn_model(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device)

    if "model_state_dict" in checkpoint:
        params = checkpoint["params"]
        model = CNN(
            base_channels=params["base_channels"],
            dropout_feat=params["dropout_feat"],
            dropout=params["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = CNN(
            base_channels=64,
            dropout_feat=0.2,
            dropout=0.0,
        )
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def preprocess_qr_for_cnn(image: Image.Image) -> torch.Tensor:
    image = image.convert("L").resize((69, 69))
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def prepare_url_features(url: str, feature_columns):
    features_df = extract_url_features(url)

    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0

    features_df = features_df[feature_columns]
    return features_df


def predict_url_model_probabilities(url: str, models: dict, feature_columns):
    features_df = prepare_url_features(url, feature_columns)

    probs = {
        "lr": float(models["lr"].predict_proba(features_df)[0][1]),
        "rf": float(models["rf"].predict_proba(features_df)[0][1]),
        "xgb": float(models["xgb"].predict_proba(features_df)[0][1]),
        "mlp": float(models["mlp"].predict_proba(features_df)[0][1]),
    }

    return probs, features_df.iloc[0].to_dict()


@torch.no_grad()
def predict_cnn_probability(image: Image.Image, cnn_model, device: str = "cpu") -> float:
    tensor = preprocess_qr_for_cnn(image).to(device)
    logit = cnn_model(tensor)
    prob = torch.sigmoid(logit).item()
    return float(prob)


def predict_stacked_probability(url_probs: dict, cnn_prob: float, stacker) -> float:
    X_meta = np.array([[
        url_probs["lr"],
        url_probs["rf"],
        url_probs["xgb"],
        url_probs["mlp"],
        cnn_prob
    ]])
    prob = float(stacker.predict_proba(X_meta)[0][1])
    return prob