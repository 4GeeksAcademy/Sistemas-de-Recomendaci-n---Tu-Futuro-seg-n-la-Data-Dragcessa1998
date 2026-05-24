from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import (
    UrlStatsTransformer,
    classification_metrics,
    load_url_spam_dataset,
    tokenize_url,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "url_spam.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "url_spam_svm_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "url_spam_svm_metrics.json"
RANDOM_STATE = 42


def build_svm_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize_url,
                    token_pattern=None,
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=6000,
                ),
            ),
            (
                "url_stats",
                Pipeline(
                    [
                        ("stats", UrlStatsTransformer()),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", SVC()),
        ]
    )


def train_and_optimize() -> dict:
    df = load_url_spam_dataset(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["url"],
        df["is_spam"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["is_spam"],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"url": X_train, "is_spam": y_train}).to_csv(
        PROCESSED_DIR / "train.csv", index=False
    )
    pd.DataFrame({"url": X_test, "is_spam": y_test}).to_csv(
        PROCESSED_DIR / "test.csv", index=False
    )

    baseline_model = build_svm_pipeline()
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_metrics = classification_metrics(y_test, baseline_predictions)

    search_space = {
        "features__tfidf__ngram_range": [(1, 1), (1, 2)],
        "features__tfidf__min_df": [1, 2, 4],
        "classifier__C": [0.5, 1, 3, 10],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__gamma": ["scale", "auto"],
        "classifier__class_weight": [None, "balanced"],
    }
    optimized_search = GridSearchCV(
        estimator=build_svm_pipeline(),
        param_grid=search_space,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    optimized_search.fit(X_train, y_train)

    optimized_model = optimized_search.best_estimator_
    optimized_predictions = optimized_model.predict(X_test)
    optimized_metrics = classification_metrics(y_test, optimized_predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(optimized_model, MODEL_PATH)

    results = {
        "dataset": {
            "rows": int(df.shape[0]),
            "unique_urls": int(df["url"].nunique()),
            "spam": int(df["is_spam"].sum()),
            "not_spam": int((~df["is_spam"]).sum()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "baseline_svm_default_classifier": baseline_metrics,
        "optimized_svm": {
            "best_params": optimized_search.best_params_,
            "best_cv_f1": optimized_search.best_score_,
            **optimized_metrics,
        },
        "artifacts": {
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "train_path": str((PROCESSED_DIR / "train.csv").relative_to(PROJECT_ROOT)),
            "test_path": str((PROCESSED_DIR / "test.csv").relative_to(PROJECT_ROOT)),
        },
    }
    write_json(results, METRICS_PATH)
    return results


def main() -> None:
    results = train_and_optimize()
    baseline = results["baseline_svm_default_classifier"]
    optimized = results["optimized_svm"]

    print("URL spam detection training complete")
    print(f"Dataset rows: {results['dataset']['rows']}")
    print(
        "Baseline SVM - "
        f"accuracy: {baseline['accuracy']:.3f}, spam F1: {baseline['f1_spam']:.3f}"
    )
    print(
        "Optimized SVM - "
        f"accuracy: {optimized['accuracy']:.3f}, spam F1: {optimized['f1_spam']:.3f}"
    )
    print(f"Best params: {optimized['best_params']}")
    print(f"Saved model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
