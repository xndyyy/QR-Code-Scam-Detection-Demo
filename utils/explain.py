import pandas as pd
import xgboost as xgb


def build_feature_row_from_dict(feature_dict, feature_columns):
    row = {col: feature_dict.get(col, 0) for col in feature_columns}
    return pd.DataFrame([row])


def get_xgb_local_contributions(xgb_model, x_row, top_n=5):
    """
    Uses XGBoost's built-in pred_contribs to get local per-feature contributions
    for one specific input row.

    Positive contribution -> pushes toward phishing
    Negative contribution -> pushes toward benign
    """
    dmatrix = xgb.DMatrix(x_row)
    contribs = xgb_model.get_booster().predict(dmatrix, pred_contribs=True)

    contrib_row = contribs[0]

    # Last column is the bias term
    feature_contribs = contrib_row[:-1]
    bias = contrib_row[-1]

    explanation_df = pd.DataFrame({
        "feature": x_row.columns,
        "value": x_row.iloc[0].values,
        "contribution": feature_contribs
    })

    explanation_df["abs_contribution"] = explanation_df["contribution"].abs()
    explanation_df = explanation_df.sort_values("abs_contribution", ascending=False).head(top_n)

    return explanation_df, bias


def get_contribution_strength(contribution):
    abs_val = abs(contribution)
    if abs_val >= 1.0:
        return "Strong"
    elif abs_val >= 0.4:
        return "Moderate"
    else:
        return "Weak"


def describe_xgb_contribution(feature, value, contribution):
    direction = "toward phishing" if contribution > 0 else "toward benign"

    def phrasing(phishing_text, benign_text):
        return phishing_text if contribution > 0 else benign_text

    templates = {

        # LENGTH / STRUCTURE
        "url_length": phrasing(
            f"The URL is long ({value}), which can hide malicious intent and is commonly seen in phishing.",
            f"The URL length ({value}) is relatively normal and does not suggest obfuscation."
        ),

        "domain_length": phrasing(
            f"The domain length ({value}) is relatively long, which may indicate attempts to mimic legitimate domains.",
            f"The domain length ({value}) appears normal and not indicative of deception."
        ),

        "path_length": phrasing(
            f"The path length ({value}) is long, which may indicate hidden or complex routing.",
            f"The path length ({value}) is short and appears structurally simple."
        ),

        "query_length": phrasing(
            f"The query string length ({value}) is long, which may indicate tracking or hidden parameters.",
            f"The query string length ({value}) is short, suggesting minimal hidden parameters."
        ),

        # BOOLEAN FLAGS
        "has_path": phrasing(
            "The presence of a path suggests deeper navigation which may be used to obscure intent.",
            "The absence or simplicity of a path suggests a more straightforward URL."
        ),

        "has_query": phrasing(
            "The presence of query parameters may indicate tracking or dynamic manipulation.",
            "The absence of query parameters suggests a simpler and more transparent URL."
        ),

        "has_ip": phrasing(
            "The URL uses an IP address directly, which is often a strong phishing indicator.",
            "The URL does not use an IP address, which is less suspicious."
        ),

        # TOKEN / STRUCTURE COMPLEXITY
        "num_tokens": phrasing(
            f"The URL contains {value} tokens, indicating a complex structure often used in phishing.",
            f"The URL contains {value} tokens, suggesting a relatively simple structure."
        ),

        "domain_token_count": phrasing(
            f"The domain is split into {value} parts, which may be used to imitate legitimate sites.",
            f"The domain structure ({value} parts) appears normal."
        ),

        "longest_token_length": phrasing(
            f"The longest token length ({value}) is high, which may indicate encoded or obfuscated content.",
            f"The longest token length ({value}) does not strongly indicate suspicious encoding."
        ),

        "avg_token_length": phrasing(
            f"The average token length ({value:.2f}) is high, suggesting unusual URL construction.",
            f"The average token length ({value:.2f}) appears normal."
        ),

        # ENTROPY
        "domain_entropy": phrasing(
            f"The domain entropy ({value:.2f}) is high, indicating a less readable or more random-looking domain.",
            f"The domain entropy ({value:.2f}) does not indicate an irregular pattern."
        ),

        # CHARACTER COUNTS
        "num_digits": phrasing(
            f"The URL contains {value} digits, which may indicate obfuscation or artificial patterns.",
            f"The number of digits ({value}) is not unusually high."
        ),

        "num_letters": phrasing(
            f"The URL contains {value} letters, contributing to its overall structure.",
            f"The number of letters ({value}) appears typical."
        ),

        "num_special": phrasing(
            f"The URL contains {value} special characters, which may indicate obfuscation.",
            f"The number of special characters ({value}) is relatively low."
        ),

        # SPECIFIC SYMBOLS
        "num_slash": phrasing(
            f"The URL contains {value} slashes, suggesting deeper navigation paths.",
            f"The number of slashes ({value}) appears normal."
        ),

        "num_dot": phrasing(
            f"The URL contains {value} dots, which may indicate complex or deceptive structure.",
            f"The number of dots ({value}) is not unusually high."
        ),

        "num_hyphen": phrasing(
            f"The URL contains {value} hyphens, which may be used to mimic legitimate domains.",
            f"The number of hyphens ({value}) is not suspicious."
        ),

        "num_at": phrasing(
            f"The presence of '@' ({value}) can obscure the true destination of the URL.",
            f"No suspicious '@' usage detected."
        ),

        "num_equal": phrasing(
            f"The URL contains {value} '=' signs, suggesting parameter manipulation.",
            f"The number of '=' signs ({value}) is not unusual."
        ),

        "num_qmark": phrasing(
            f"The URL contains {value} '?' characters, indicating query-based complexity.",
            f"The number of '?' characters ({value}) is minimal."
        ),

        "num_percent": phrasing(
            f"The URL contains {value} '%' encodings, which may indicate encoded or hidden content.",
            f"The number of encoded characters ({value}) is low."
        ),

        "num_semicolon": phrasing(
            f"The URL contains {value} semicolons, which are uncommon and may indicate manipulation.",
            f"The number of semicolons ({value}) is negligible."
        ),

        "num_ampersand": phrasing(
            f"The URL contains {value} '&' symbols, indicating multiple parameters.",
            f"The number of parameters ({value}) appears limited."
        ),

        # RATIOS
        "digit_ratio": phrasing(
            f"The digit ratio ({value:.2f}) is high, suggesting artificial or machine-generated patterns.",
            f"The digit ratio ({value:.2f}) is within a normal range."
        ),

        "special_ratio": phrasing(
            f"The special character ratio ({value:.2f}) is high, indicating potential obfuscation.",
            f"The special character ratio ({value:.2f}) is low and not suspicious."
        ),
    }

    return templates.get(
        feature,
        f"{feature} = {value}. This contributed {direction}."
    )

FEATURE_LABELS = {
    "url_length": "URL Length",
    "domain_length": "Domain Length",
    "num_subdomains": "Subdomain Count",
    "path_length": "Path Length",
    "has_path": "Has Path",
    "query_length": "Query Length",
    "has_query": "Has Query",
    "num_tokens": "Number of Tokens",
    "domain_token_count": "Domain Token Count",
    "longest_token_length": "Longest Token Length",
    "avg_token_length": "Average Token Length",
    "domain_entropy": "Domain Entropy",
    "num_digits": "Digit Count",
    "num_letters": "Letter Count",
    "num_special": "Special Character Count",
    "num_slash": "Number of '/' Characters",
    "num_dot": "Number of '.' Characters",
    "num_hyphen": "Number of '-' Characters",
    "num_at": "Number of '@' Characters",
    "num_equal": "Number of '=' Characters",
    "num_qmark": "Number of '?' Characters",
    "num_percent": "Number of '%' Characters",
    "num_semicolon": "Number of ';' Characters",
    "num_ampersand": "Number of '&' Characters",
    "has_ip": "Uses IP Address",
    "digit_ratio": "Digit Ratio",
    "special_ratio": "Special Character Ratio",
}