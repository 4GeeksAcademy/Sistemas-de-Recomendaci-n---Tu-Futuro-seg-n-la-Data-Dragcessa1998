from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


URL_STOPWORDS = ENGLISH_STOP_WORDS.union(
    {
        "http",
        "https",
        "www",
        "com",
        "org",
        "net",
        "html",
        "php",
        "asp",
        "aspx",
        "index",
        "default",
        "utm",
        "utm_source",
        "utm_medium",
        "utm_campaign",
    }
)


def load_url_spam_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the URL spam dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download url_spam.csv into data/raw/."
        )

    df = pd.read_csv(path)
    expected_columns = {"url", "is_spam"}
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.loc[:, ["url", "is_spam"]].dropna().drop_duplicates().reset_index(drop=True)
    df["url"] = df["url"].astype(str).str.strip()
    df["is_spam"] = df["is_spam"].astype(bool)
    return df[df["url"].ne("")].reset_index(drop=True)


def simple_lemmatize(token: str) -> str:
    """Small deterministic lemmatizer for URL tokens without external corpora."""
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize_url(url: str) -> list[str]:
    """Split URLs by punctuation, remove stopwords and normalize tokens."""
    parsed = urlparse(str(url).lower())
    query_tokens = [item for pair in parse_qsl(parsed.query) for item in pair]
    raw_text = " ".join(
        [
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.fragment,
            " ".join(query_tokens),
        ]
    )
    raw_text = unquote(raw_text)
    tokens = re.split(r"[^a-z0-9]+", raw_text)

    clean_tokens = []
    for token in tokens:
        token = simple_lemmatize(token.strip())
        if len(token) < 2 or token in URL_STOPWORDS:
            continue
        clean_tokens.append(token)
    return clean_tokens


class UrlStatsTransformer(BaseEstimator, TransformerMixin):
    """Numerical URL features that complement TF-IDF text features."""

    feature_names = [
        "url_length",
        "digit_count",
        "letter_count",
        "special_char_count",
        "dot_count",
        "hyphen_count",
        "slash_count",
        "query_length",
        "has_https",
        "has_query",
        "subdomain_count",
    ]

    def fit(self, X: Iterable[str], y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:  # noqa: N803 - sklearn API
        rows = []
        for value in X:
            url = str(value)
            parsed = urlparse(url)
            netloc_parts = [part for part in parsed.netloc.split(".") if part]
            rows.append(
                [
                    len(url),
                    sum(char.isdigit() for char in url),
                    sum(char.isalpha() for char in url),
                    sum(not char.isalnum() for char in url),
                    url.count("."),
                    url.count("-"),
                    url.count("/"),
                    len(parsed.query),
                    int(parsed.scheme == "https"),
                    int(bool(parsed.query)),
                    max(len(netloc_parts) - 2, 0),
                ]
            )
        return np.asarray(rows, dtype=float)


def classification_metrics(y_true, y_pred) -> dict:
    """Return compact metrics plus the full classification report."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_spam": precision_score(y_true, y_pred, pos_label=True, zero_division=0),
        "recall_spam": recall_score(y_true, y_pred, pos_label=True, zero_division=0),
        "f1_spam": f1_score(y_true, y_pred, pos_label=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["not_spam", "spam"],
            output_dict=True,
            zero_division=0,
        ),
    }


def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
