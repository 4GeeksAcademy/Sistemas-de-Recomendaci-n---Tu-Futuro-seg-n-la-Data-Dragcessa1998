from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "income_high"
ORIGINAL_TARGET = "income"
NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
SENSITIVE_FEATURES = {"sex", "race", "native_country", "relationship"}
RECOMMENDABLE_FEATURES = {"education", "education_num", "occupation", "workclass", "hours_per_week"}


EDUCATION_PATH = [
    ("HS-grad", 9),
    ("Some-college", 10),
    ("Assoc-voc", 11),
    ("Assoc-acdm", 12),
    ("Bachelors", 13),
    ("Masters", 14),
    ("Prof-school", 15),
    ("Doctorate", 16),
]
CAREER_TRACKS = ["Exec-managerial", "Prof-specialty", "Tech-support", "Sales"]
WORKCLASS_TRACKS = ["Private", "Federal-gov", "Self-emp-inc"]
HOURS_TRACKS = [40, 45, 50]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        column.strip().lower().replace(".", "_").replace("-", "_")
        for column in df.columns
    ]
    return df


def load_adult_income_dataset(path: str | Path) -> pd.DataFrame:
    """Load and clean the Adult Census Income dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download adult-census-income.csv into data/raw/."
        )

    df = normalize_columns(pd.read_csv(path))
    if ORIGINAL_TARGET not in df.columns:
        raise ValueError(f"Missing required target column: {ORIGINAL_TARGET}")

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})

    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[ORIGINAL_TARGET].str.replace(".", "", regex=False).eq(">50K")

    categorical_features = get_categorical_features(df)
    for column in categorical_features:
        df[column] = df[column].fillna("Unknown")

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=NUMERIC_FEATURES + [TARGET]).reset_index(drop=True)
    return df


def get_categorical_features(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in NUMERIC_FEATURES + [ORIGINAL_TARGET, TARGET]
    ]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return NUMERIC_FEATURES + get_categorical_features(df)


def build_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_classifier(y_true, y_pred, y_score) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_high_income": precision_score(y_true, y_pred, zero_division=0),
        "recall_high_income": recall_score(y_true, y_pred, zero_division=0),
        "f1_high_income": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["<=50K", ">50K"],
            output_dict=True,
            zero_division=0,
        ),
    }


def make_default_profile(training_data: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in feature_columns:
        if column in NUMERIC_FEATURES:
            defaults[column] = float(training_data[column].median())
        else:
            defaults[column] = training_data[column].mode(dropna=True).iloc[0]
    return defaults


def profile_to_frame(profile: dict[str, Any], defaults: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    complete = defaults.copy()
    complete.update({key: value for key, value in profile.items() if key in feature_columns})
    return pd.DataFrame([complete], columns=feature_columns)


@dataclass
class RecommendationResult:
    base_probability: float
    recommendations: list[dict[str, Any]]
    similar_successful_profiles: list[dict[str, Any]]


class IncomeRecommendationSystem:
    """Hybrid recommender: model uplift plus content-based successful neighbors."""

    def __init__(self, model: Pipeline, training_data: pd.DataFrame, feature_columns: list[str]):
        self.model = model
        self.feature_columns = feature_columns
        self.defaults = make_default_profile(training_data, feature_columns)
        self.high_income_profiles = (
            training_data.loc[training_data[TARGET], feature_columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.neighbors = NearestNeighbors(n_neighbors=min(5, len(self.high_income_profiles)), metric="cosine")
        high_income_matrix = self.model.named_steps["preprocessor"].transform(self.high_income_profiles)
        self.neighbors.fit(high_income_matrix)

    def predict_probability(self, profile: dict[str, Any]) -> float:
        frame = profile_to_frame(profile, self.defaults, self.feature_columns)
        return float(self.model.predict_proba(frame)[0, 1])

    def recommend(self, profile: dict[str, Any], top_n: int = 5) -> RecommendationResult:
        completed = profile_to_frame(profile, self.defaults, self.feature_columns).iloc[0].to_dict()
        base_probability = self.predict_probability(completed)
        candidates = self._candidate_profiles(completed)

        scored = []
        for title, candidate, rationale in candidates:
            probability = self.predict_probability(candidate)
            uplift = probability - base_probability
            if uplift <= 0:
                continue
            scored.append(
                {
                    "recommendation": title,
                    "estimated_probability": round(probability, 4),
                    "probability_uplift": round(uplift, 4),
                    "changes": {
                        key: value
                        for key, value in candidate.items()
                        if key in RECOMMENDABLE_FEATURES and completed.get(key) != value
                    },
                    "rationale": rationale,
                }
            )

        scored = sorted(scored, key=lambda item: item["probability_uplift"], reverse=True)[:top_n]
        return RecommendationResult(
            base_probability=round(base_probability, 4),
            recommendations=scored,
            similar_successful_profiles=self._similar_successful_profiles(completed),
        )

    def _candidate_profiles(self, profile: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
        candidates: list[tuple[str, dict[str, Any], str]] = []
        current_education_num = int(profile.get("education_num", 0))

        for education, education_num in EDUCATION_PATH:
            if education_num > current_education_num:
                candidate = profile.copy()
                candidate["education"] = education
                candidate["education_num"] = education_num
                candidates.append(
                    (
                        f"Avanzar a {education}",
                        candidate,
                        "La educacion suele aumentar el acceso a ocupaciones mejor remuneradas.",
                    )
                )

        for occupation in CAREER_TRACKS:
            if profile.get("occupation") != occupation:
                candidate = profile.copy()
                candidate["occupation"] = occupation
                candidates.append(
                    (
                        f"Orientar trayectoria hacia {occupation}",
                        candidate,
                        "Se simula una transicion hacia una ocupacion asociada a mayor ingreso en el dataset.",
                    )
                )

        current_hours = int(profile.get("hours_per_week", 0))
        for hours in HOURS_TRACKS:
            if hours > current_hours:
                candidate = profile.copy()
                candidate["hours_per_week"] = hours
                candidates.append(
                    (
                        f"Aumentar disponibilidad a {hours} horas semanales",
                        candidate,
                        "El modelo evalua si una mayor disponibilidad laboral mejora la probabilidad estimada.",
                    )
                )

        for workclass in WORKCLASS_TRACKS:
            if profile.get("workclass") != workclass:
                candidate = profile.copy()
                candidate["workclass"] = workclass
                candidates.append(
                    (
                        f"Explorar workclass {workclass}",
                        candidate,
                        "Se compara el perfil contra categorias laborales frecuentes en ingresos altos.",
                    )
                )

        return candidates

    def _similar_successful_profiles(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        frame = profile_to_frame(profile, self.defaults, self.feature_columns)
        transformed = self.model.named_steps["preprocessor"].transform(frame)
        distances, indices = self.neighbors.kneighbors(transformed)

        neighbors = []
        for distance, index in zip(distances[0], indices[0]):
            row = self.high_income_profiles.iloc[int(index)]
            neighbors.append(
                {
                    "similarity": round(1 - float(distance), 4),
                    "education": row.get("education"),
                    "occupation": row.get("occupation"),
                    "workclass": row.get("workclass"),
                    "hours_per_week": int(row.get("hours_per_week")),
                }
            )
        return neighbors


def simulated_profiles() -> dict[str, dict[str, Any]]:
    return {
        "young_part_time_hs": {
            "age": 25,
            "workclass": "Private",
            "fnlwgt": 180000,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 25,
            "native_country": "United-States",
        },
        "mid_career_college_sales": {
            "age": 38,
            "workclass": "Private",
            "fnlwgt": 200000,
            "education": "Some-college",
            "education_num": 10,
            "marital_status": "Married-civ-spouse",
            "occupation": "Sales",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        },
        "adult_service_worker": {
            "age": 45,
            "workclass": "Private",
            "fnlwgt": 190000,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Divorced",
            "occupation": "Other-service",
            "relationship": "Unmarried",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 35,
            "native_country": "United-States",
        },
    }


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
