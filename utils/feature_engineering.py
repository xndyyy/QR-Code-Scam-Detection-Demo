import math
import re
import pandas as pd
from urllib.parse import urlparse


def normalize_url(url):
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url


def extract_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None


def extract_path(url):
    try:
        return urlparse(url).path
    except Exception:
        return None


def extract_query(url):
    try:
        return urlparse(url).query
    except Exception:
        return None


def tokenize_url(url):
    if not isinstance(url, str):
        return []
    return re.split(r"[/\-,._=?&]+", url)


def longest_token(url):
    if not isinstance(url, str):
        return 0
    tokens = re.split(r"[\/\.\-\_\?\=\&\:\%]+", url)
    tokens = [t for t in tokens if t]
    return max((len(t) for t in tokens), default=0)


def avg_token_length(url):
    if not isinstance(url, str):
        return 0
    tokens = re.split(r"[\/\.\-\_\?\=\&\:\%]+", url)
    tokens = [t for t in tokens if t]
    if len(tokens) == 0:
        return 0
    return sum(len(t) for t in tokens) / len(tokens)


def shannon_entropy(s):
    if not isinstance(s, str) or len(s) == 0:
        return 0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def count_char(url, ch):
    try:
        return url.count(ch)
    except Exception:
        return 0


def has_ip(url):
    if not isinstance(url, str):
        return 0
    return int(bool(re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", url)))


def extract_url_features(decoded_url: str) -> pd.DataFrame:
    """
    Recreates the URL feature engineering used in the training notebook
    as closely as possible for a single decoded URL.
    """
    normalized_url = normalize_url(decoded_url)
    domain = extract_domain(normalized_url)
    path = extract_path(normalized_url)
    query = extract_query(normalized_url)
    tokens = tokenize_url(decoded_url)

    url_length = len(decoded_url) if isinstance(decoded_url, str) else 0
    num_digits = len(re.findall(r"\d", decoded_url)) if isinstance(decoded_url, str) else 0
    num_letters = len(re.findall(r"[A-Za-z]", decoded_url)) if isinstance(decoded_url, str) else 0
    num_special = len(re.findall(r"[^A-Za-z0-9]", decoded_url)) if isinstance(decoded_url, str) else 0

    features = {
        "url_length": url_length,
        "domain_length": len(domain) if isinstance(domain, str) else 0,
        "num_subdomains": domain.count(".") if isinstance(domain, str) else 0,
        "path_length": len(path) if isinstance(path, str) else 0,
        "has_path": int(len(path) > 1) if isinstance(path, str) else 0,
        "query_length": len(query) if isinstance(query, str) else 0,
        "has_query": int(len(query) > 0) if isinstance(query, str) else 0,
        "num_tokens": len(tokens) if isinstance(tokens, list) else 0,
        "domain_token_count": len(domain.split(".")) if isinstance(domain, str) else 0,
        "longest_token_length": longest_token(decoded_url),
        "avg_token_length": avg_token_length(decoded_url),
        "domain_entropy": shannon_entropy(domain),
        "num_digits": num_digits,
        "num_letters": num_letters,
        "num_special": num_special,
        "num_slash": count_char(decoded_url, "/"),
        "num_dot": count_char(decoded_url, "."),
        "num_hyphen": count_char(decoded_url, "-"),
        "num_at": count_char(decoded_url, "@"),
        "num_equal": count_char(decoded_url, "="),
        "num_qmark": count_char(decoded_url, "?"),
        "num_percent": count_char(decoded_url, "%"),
        "num_semicolon": count_char(decoded_url, ";"),
        "num_ampersand": count_char(decoded_url, "&"),
        "has_ip": has_ip(decoded_url),
        "digit_ratio": (num_digits / url_length) if url_length > 0 else 0,
        "special_ratio": (num_special / url_length) if url_length > 0 else 0,
    }

    return pd.DataFrame([features])