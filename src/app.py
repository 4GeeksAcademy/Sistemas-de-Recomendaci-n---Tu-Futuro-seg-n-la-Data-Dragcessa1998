from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from utils import (
    TARGET,
    IncomeRecommendationSystem,
    build_preprocessor,
    evaluate_classifier,
    feature_importance_report,
    get_categorical_features,
    get_feature_columns,
    load_adult_income_dataset,
    simulated_profiles,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult-census-income.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "adult_income_recommender.joblib"
METRICS_PATH = MODEL_DIR / "adult_income_metrics.json"
RECOMMENDATIONS_PATH = MODEL_DIR / "sample_recommendations.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RANDOM_STATE = 42


def build_model(categorical_features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(categorical_features)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_model_and_recommender() -> dict:
    df = load_adult_income_dataset(RAW_DATA_PATH)
    feature_columns = get_feature_columns(df)
    categorical_features = get_categorical_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_columns],
        df[TARGET],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    baseline_model = build_model(categorical_features)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_score = baseline_model.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_classifier(y_test, baseline_pred, baseline_score)

    search = GridSearchCV(
        estimator=build_model(categorical_features),
        param_grid={
            "classifier__C": [0.3, 1.0, 3.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=5,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    optimized_pred = best_model.predict(X_test)
    optimized_score = best_model.predict_proba(X_test)[:, 1]
    optimized_metrics = evaluate_classifier(y_test, optimized_pred, optimized_score)
    importance = feature_importance_report(best_model, categorical_features)

    recommender = IncomeRecommendationSystem(
        model=best_model,
        training_data=train_df,
        feature_columns=feature_columns,
    )
    sample_recommendations = {
        name: recommender.recommend(profile).__dict__
        for name, profile in simulated_profiles().items()
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "recommender": recommender,
            "feature_columns": feature_columns,
            "target": TARGET,
        },
        MODEL_PATH,
    )

    results = {
        "problem_definition": {
            "what_is_recommended": "Trayectorias accionables de educacion, ocupacion, tipo de trabajo y horas semanales.",
            "user_definition": "Una persona adulta representada por su perfil demografico y socioeconomico.",
            "profile_variables": feature_columns,
            "approach": "Sistema hibrido: clasificador supervisado para estimar probabilidad de >50K y vecinos similares de alto ingreso para filtrado basado en contenido.",
        },
        "dataset": {
            "rows_after_cleaning": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "high_income": int(df[TARGET].sum()),
            "low_income": int((~df[TARGET]).sum()),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
        },
        "baseline_classifier": baseline_metrics,
        "optimized_classifier": {
            "best_params": search.best_params_,
            "best_cv_roc_auc": search.best_score_,
            **optimized_metrics,
        },
        "artifacts": {
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "recommendations_path": str(RECOMMENDATIONS_PATH.relative_to(PROJECT_ROOT)),
            "feature_importance_path": str(FEATURE_IMPORTANCE_PATH.relative_to(PROJECT_ROOT)),
            "train_path": str((PROCESSED_DIR / "train.csv").relative_to(PROJECT_ROOT)),
            "test_path": str((PROCESSED_DIR / "test.csv").relative_to(PROJECT_ROOT)),
            "figures_dir": str(FIGURES_DIR.relative_to(PROJECT_ROOT)),
        },
    }

    write_json(results, METRICS_PATH)
    write_json(sample_recommendations, RECOMMENDATIONS_PATH)
    write_json(importance, FEATURE_IMPORTANCE_PATH)
    create_visualizations(df, importance)
    return results


def create_visualizations(df: pd.DataFrame, importance: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    distribution = (
        df["income_high"]
        .map({False: "<=50K", True: ">50K"})
        .value_counts()
        .rename_axis("income")
        .reset_index(name="count")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=distribution, x="income", y="count", hue="income", palette=["#4d5f85", "#d05a3f"], legend=False)
    plt.title("Distribucion de ingresos")
    plt.xlabel("Ingreso anual")
    plt.ylabel("Personas")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "income_distribution.png", dpi=160)
    plt.close()

    grouped = pd.DataFrame(importance["grouped_importance"]).head(12)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=grouped, y="feature", x="total_abs_coefficient", color="#4d5f85")
    plt.title("Variables con mayor peso en el modelo")
    plt.xlabel("Suma de coeficientes absolutos")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_feature_importance.png", dpi=160)
    plt.close()

    social = pd.DataFrame(importance["social_variable_importance"])
    if not social.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=social, y="feature", x="total_abs_coefficient", color="#d05a3f")
        plt.title("Peso de variables sociales")
        plt.xlabel("Suma de coeficientes absolutos")
        plt.ylabel("Variable social")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "social_variable_importance.png", dpi=160)
        plt.close()


def main() -> None:
    results = train_model_and_recommender()
    baseline = results["baseline_classifier"]
    optimized = results["optimized_classifier"]

    print("Adult Income recommendation project complete")
    print(f"Rows after cleaning: {results['dataset']['rows_after_cleaning']}")
    print(
        "Baseline classifier - "
        f"accuracy: {baseline['accuracy']:.3f}, ROC AUC: {baseline['roc_auc']:.3f}"
    )
    print(
        "Optimized classifier - "
        f"accuracy: {optimized['accuracy']:.3f}, ROC AUC: {optimized['roc_auc']:.3f}"
    )
    print(f"Best params: {optimized['best_params']}")
    print(f"Saved model and recommender: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved sample recommendations: {RECOMMENDATIONS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
