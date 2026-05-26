from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from xgboost import XGBClassifier

from .validation import validate_dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
NOTEBOOK_DIR = ROOT / "notebooks"
MODELS_DIR = ROOT / "models"
VISUALS_DIR = ROOT / "visuals"
REPORTS_DIR = ROOT / "reports"
METRICS_DIR = ROOT / "metrics"
SRC_DIR = ROOT / "src"
DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality.json"
MISSING_VALUE_REPORT = REPORTS_DIR / "missing_value_profile.json"
CATEGORICAL_PROFILE_REPORT = REPORTS_DIR / "categorical_profile.json"
NORMALIZED_BEHAVIOR_REPORT = REPORTS_DIR / "normalized_behavior_profile.json"
SESSION_ANALYSIS_REPORT = REPORTS_DIR / "session_analysis.json"
CUSTOMER_SEGMENT_REPORT = REPORTS_DIR / "customer_segments.json"
BEHAVIOR_SCALE_REPORT = REPORTS_DIR / "behavior_scale_profile.json"
OUTLIER_REPORT = REPORTS_DIR / "outlier_profile.json"

RANDOM_STATE = 42
TARGET = "buying_mood"
MOODS = [
    "Impulsive Buyer",
    "Luxury Shopper",
    "Budget Conscious",
    "Window Shopper",
    "Emotional Buyer",
]

NUMERIC_FEATURES = [
    "age",
    "income",
    "browsing_time",
    "cart_items",
    "purchase_frequency",
    "discount_usage",
    "avg_spending",
    "late_night_activity",
    "impulsive_clicks",
    "impulse_score",
    "spending_efficiency",
    "shopping_intensity",
    "engagement_score",
    "conversion_probability",
    "browsing_pressure",
    "discount_dependency",
    "emotional_purchase_index",
]
RAW_NUMERIC_FEATURES = [
    "age",
    "income",
    "browsing_time",
    "cart_items",
    "purchase_frequency",
    "discount_usage",
    "avg_spending",
    "late_night_activity",
    "impulsive_clicks",
]
CATEGORICAL_FEATURES = ["gender", "preferred_category"]


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_accuracy: float
    estimator: Any
    predictions: np.ndarray



def ensure_directories() -> None:
    for directory in [DATA_DIR, NOTEBOOK_DIR, MODELS_DIR, VISUALS_DIR, REPORTS_DIR, METRICS_DIR, SRC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)



def generate_synthetic_dataset(n_rows: int = 2400, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genders = np.array(["Female", "Male", "Non-binary"])
    categories = np.array(["Fashion", "Electronics", "Beauty", "Home", "Groceries", "Luxury"])

    buying_mood = rng.choice(MOODS, n_rows, p=[0.21, 0.19, 0.23, 0.18, 0.19])

    profiles = {
        "Impulsive Buyer": {
            "age": (29, 7),
            "income": (54000, 14000),
            "browsing_time": (31, 7),
            "cart_items": (7, 2),
            "purchase_frequency": (4.8, 1.6),
            "discount_usage": (34, 16),
            "avg_spending": (165, 60),
            "late_night_activity": (72, 12),
            "impulsive_clicks": (12, 3),
            "preferred_category": (["Fashion", "Electronics"], [0.62, 0.38]),
        },
        "Luxury Shopper": {
            "age": (41, 8),
            "income": (121000, 18000),
            "browsing_time": (18, 6),
            "cart_items": (4, 2),
            "purchase_frequency": (8.5, 2.2),
            "discount_usage": (15, 9),
            "avg_spending": (395, 85),
            "late_night_activity": (24, 10),
            "impulsive_clicks": (4, 2),
            "preferred_category": (["Luxury", "Electronics"], [0.7, 0.3]),
        },
        "Budget Conscious": {
            "age": (35, 9),
            "income": (44000, 9000),
            "browsing_time": (20, 6),
            "cart_items": (3, 1),
            "purchase_frequency": (9.2, 2.4),
            "discount_usage": (79, 11),
            "avg_spending": (92, 28),
            "late_night_activity": (18, 8),
            "impulsive_clicks": (3, 1),
            "preferred_category": (["Groceries", "Home"], [0.68, 0.32]),
        },
        "Window Shopper": {
            "age": (28, 7),
            "income": (62000, 13000),
            "browsing_time": (43, 8),
            "cart_items": (1, 1),
            "purchase_frequency": (2.8, 1.0),
            "discount_usage": (30, 14),
            "avg_spending": (48, 20),
            "late_night_activity": (26, 10),
            "impulsive_clicks": (2, 1),
            "preferred_category": (["Fashion", "Beauty"], [0.55, 0.45]),
        },
        "Emotional Buyer": {
            "age": (32, 8),
            "income": (58000, 12000),
            "browsing_time": (27, 7),
            "cart_items": (5, 2),
            "purchase_frequency": (4.2, 1.5),
            "discount_usage": (43, 15),
            "avg_spending": (185, 55),
            "late_night_activity": (83, 9),
            "impulsive_clicks": (9, 3),
            "preferred_category": (["Beauty", "Fashion"], [0.6, 0.4]),
        },
    }

    age = np.zeros(n_rows)
    gender = rng.choice(genders, n_rows, p=[0.46, 0.47, 0.07])
    income = np.zeros(n_rows)
    browsing_time = np.zeros(n_rows)
    cart_items = np.zeros(n_rows)
    purchase_frequency = np.zeros(n_rows)
    discount_usage = np.zeros(n_rows)
    avg_spending = np.zeros(n_rows)
    preferred_category = np.empty(n_rows, dtype=object)
    late_night_activity = np.zeros(n_rows)
    impulsive_clicks = np.zeros(n_rows)

    for mood in MOODS:
        mask = buying_mood == mood
        count = int(mask.sum())
        profile = profiles[mood]
        age[mask] = np.clip(rng.normal(*profile["age"], count), 18, 67)
        income[mask] = np.clip(rng.normal(*profile["income"], count), 18000, 180000)
        browsing_time[mask] = np.clip(rng.normal(*profile["browsing_time"], count), 2, 70)
        cart_items[mask] = np.clip(rng.poisson(profile["cart_items"][0], count) + rng.integers(0, profile["cart_items"][1] + 1, count), 0, 18)
        purchase_frequency[mask] = np.clip(rng.normal(*profile["purchase_frequency"], count), 0.2, 24)
        discount_usage[mask] = np.clip(rng.normal(*profile["discount_usage"], count), 0, 100)
        avg_spending[mask] = np.clip(rng.normal(*profile["avg_spending"], count) + 0.0006 * income[mask], 15, 800)
        late_night_activity[mask] = np.clip(rng.normal(*profile["late_night_activity"], count), 0, 100)
        impulsive_clicks[mask] = np.clip(rng.normal(*profile["impulsive_clicks"], count), 0, 36)
        preferred_category[mask] = rng.choice(profile["preferred_category"][0], count, p=profile["preferred_category"][1])

        if mood == "Impulsive Buyer":
            browsing_time[mask] = np.clip(browsing_time[mask] + rng.normal(0, 3.4, count), 2, 70)
            cart_items[mask] = np.clip(cart_items[mask] + rng.normal(0, 1.2, count), 0, 18)
            late_night_activity[mask] = np.clip(late_night_activity[mask] + rng.normal(0, 5.5, count), 0, 100)
            impulsive_clicks[mask] = np.clip(impulsive_clicks[mask] + rng.normal(0, 1.5, count), 0, 36)
        elif mood == "Window Shopper":
            browsing_time[mask] = np.clip(browsing_time[mask] + rng.normal(0, 4.0, count), 2, 70)
            cart_items[mask] = np.clip(cart_items[mask] + rng.normal(0, 1.1, count), 0, 18)
            purchase_frequency[mask] = np.clip(purchase_frequency[mask] + rng.normal(0, 0.8, count), 0.2, 24)
            late_night_activity[mask] = np.clip(late_night_activity[mask] + rng.normal(0, 3.5, count), 0, 100)
        elif mood == "Emotional Buyer":
            late_night_activity[mask] = np.clip(late_night_activity[mask] + rng.normal(0, 6.0, count), 0, 100)
            impulsive_clicks[mask] = np.clip(impulsive_clicks[mask] + rng.normal(0, 2.0, count), 0, 36)
            cart_items[mask] = np.clip(cart_items[mask] + rng.normal(0, 1.0, count), 0, 18)

    age = np.clip(age + rng.normal(0, 2.0, n_rows), 18, 67)
    income = np.clip(income + rng.normal(0, 4500, n_rows), 18000, 180000)
    browsing_time = np.clip(browsing_time + rng.normal(0, 2.6, n_rows), 2, 70)
    cart_items = np.clip(cart_items + rng.normal(0, 0.8, n_rows), 0, 18)
    purchase_frequency = np.clip(purchase_frequency + rng.normal(0, 0.7, n_rows), 0.2, 24)
    discount_usage = np.clip(discount_usage + rng.normal(0, 5.5, n_rows), 0, 100)
    avg_spending = np.clip(avg_spending + rng.normal(0, 18, n_rows), 15, 800)
    late_night_activity = np.clip(late_night_activity + rng.normal(0, 4.0, n_rows), 0, 100)
    impulsive_clicks = np.clip(impulsive_clicks + rng.normal(0, 1.0, n_rows), 0, 36)

    cart_items = np.round(cart_items).astype(int)
    impulsive_clicks = np.round(impulsive_clicks).astype(int)

    impulse_score = 0.32 * impulsive_clicks + 0.18 * cart_items + 0.22 * late_night_activity / 10 + 0.12 * browsing_time - 0.05 * purchase_frequency
    spending_efficiency = avg_spending / np.maximum(cart_items + 1, 1)
    shopping_intensity = 0.35 * browsing_time + 0.26 * cart_items + 0.20 * impulsive_clicks + 0.12 * purchase_frequency
    browsing_pressure = 0.45 * browsing_time + 0.30 * impulsive_clicks + 0.15 * late_night_activity
    discount_dependency = 0.70 * discount_usage + 0.12 * np.maximum(55 - purchase_frequency, 0)
    emotional_purchase_index = 0.40 * late_night_activity + 0.28 * impulsive_clicks + 0.18 * cart_items + 0.08 * browsing_time

    noisy_label_mask = rng.random(n_rows) < 0.04
    if noisy_label_mask.any():
        alternative_moods = np.array(MOODS, dtype=object)
        for index in np.where(noisy_label_mask)[0]:
            current_mood = buying_mood[index]
            other_moods = alternative_moods[alternative_moods != current_mood]
            buying_mood[index] = rng.choice(other_moods)

    dataframe = pd.DataFrame(
        {
            "user_id": [f"U{idx:05d}" for idx in range(1, n_rows + 1)],
            "age": age,
            "gender": gender,
            "income": income.round(2),
            "browsing_time": browsing_time.round(2),
            "cart_items": cart_items,
            "purchase_frequency": purchase_frequency.round(2),
            "discount_usage": discount_usage.round(2),
            "avg_spending": avg_spending.round(2),
            "preferred_category": preferred_category,
            "late_night_activity": late_night_activity.round(2),
            "impulsive_clicks": impulsive_clicks,
            TARGET: buying_mood,
        }
    )

    return dataframe



def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    cleaned = cleaned.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    cleaned["gender"] = cleaned["gender"].replace({"nonbinary": "Non-binary"})
    cleaned["preferred_category"] = cleaned["preferred_category"].fillna("Fashion")
    cleaned[RAW_NUMERIC_FEATURES] = cleaned[RAW_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    return cleaned



def normalize_categorical_fields(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    gender_map = {"f": "Female", "m": "Male", "nb": "Non-binary", "nonbinary": "Non-binary"}
    normalized["gender"] = normalized["gender"].astype(str).str.strip().str.lower().replace(gender_map).str.title().replace({"Non-Binary": "Non-binary"})
    normalized["preferred_category"] = normalized["preferred_category"].astype(str).str.strip().str.title()
    canonical_categories = ["Beauty", "Electronics", "Fashion", "Groceries", "Home", "Luxury"]
    normalized["preferred_category"] = pd.Categorical(normalized["preferred_category"], categories=canonical_categories, ordered=True)
    return normalized



def summarize_categorical_distribution(dataframe: pd.DataFrame) -> dict[str, Any]:
    return {
        "gender_distribution": dataframe["gender"].value_counts().to_dict(),
        "preferred_category_distribution": dataframe["preferred_category"].astype(str).value_counts().to_dict(),
    }



def add_missing_values(dataframe: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    with_missing = dataframe.copy()
    for column, rate in {"income": 0.03, "discount_usage": 0.035, "preferred_category": 0.02, "avg_spending": 0.025}.items():
        mask = rng.random(len(with_missing)) < rate
        with_missing.loc[mask, column] = np.nan
    return with_missing



def summarize_missingness(dataframe: pd.DataFrame) -> dict[str, Any]:
    missing_counts = dataframe.isna().sum().to_dict()
    missing_rates = {
        column: round(float(count) / float(len(dataframe)), 4) if len(dataframe) else 0.0
        for column, count in missing_counts.items()
        if count
    }
    return {
        "row_count": int(len(dataframe)),
        "columns_with_missing": sorted([column for column, count in missing_counts.items() if count]),
        "missing_counts": missing_counts,
        "missing_rates": missing_rates,
    }



def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    engineered = dataframe.copy()
    engineered["impulse_score"] = 0.32 * engineered["impulsive_clicks"] + 0.18 * engineered["cart_items"] + 0.22 * engineered["late_night_activity"] / 10 + 0.12 * engineered["browsing_time"] - 0.05 * engineered["purchase_frequency"]
    engineered["spending_efficiency"] = engineered["avg_spending"] / (engineered["cart_items"] + 1)
    engineered["shopping_intensity"] = (
        0.28 * engineered["browsing_time"]
        + 0.22 * engineered["cart_items"]
        + 0.18 * engineered["impulsive_clicks"]
        + 0.16 * engineered["purchase_frequency"]
        + 0.10 * engineered["late_night_activity"]
        + 0.06 * (engineered["avg_spending"] / 100)
    )
    engineered["engagement_score"] = 0.40 * engineered["shopping_intensity"] + 0.30 * engineered["browsing_time"] + 0.18 * engineered["late_night_activity"] + 0.12 * engineered["impulsive_clicks"]
    conversion_base = (
        0.24 * engineered["purchase_frequency"]
        + 0.18 * engineered["cart_items"]
        + 0.16 * engineered["engagement_score"]
        + 0.14 * engineered["discount_usage"]
        + 0.10 * engineered["avg_spending"] / 100
        - 0.12 * np.maximum(25 - engineered["income"] / 5000, 0)
    )
    engineered["conversion_probability"] = 1 / (1 + np.exp(-(conversion_base - conversion_base.mean()) / 12))
    engineered["browsing_pressure"] = 0.45 * engineered["browsing_time"] + 0.30 * engineered["impulsive_clicks"] + 0.15 * engineered["late_night_activity"]
    engineered["discount_dependency"] = (
        0.62 * engineered["discount_usage"]
        + 0.14 * np.maximum(60 - engineered["purchase_frequency"] * 3, 0)
        + 0.10 * np.maximum(220 - engineered["avg_spending"], 0) / 10
        + 0.08 * np.maximum(40 - engineered["income"] / 4000, 0)
    )
    engineered["emotional_purchase_index"] = 0.40 * engineered["late_night_activity"] + 0.28 * engineered["impulsive_clicks"] + 0.18 * engineered["cart_items"] + 0.08 * engineered["browsing_time"]
    return engineered



def impute_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    imputed = dataframe.copy()
    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    imputed[RAW_NUMERIC_FEATURES] = numeric_imputer.fit_transform(imputed[RAW_NUMERIC_FEATURES])
    imputed[CATEGORICAL_FEATURES] = categorical_imputer.fit_transform(imputed[CATEGORICAL_FEATURES])
    return imputed



def normalize_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    scaler = StandardScaler()
    normalized[RAW_NUMERIC_FEATURES] = scaler.fit_transform(normalized[RAW_NUMERIC_FEATURES])
    return normalized



def normalize_behavioral_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    behavior_columns = [
        "impulse_score",
        "spending_efficiency",
        "shopping_intensity",
        "engagement_score",
        "conversion_probability",
        "browsing_pressure",
        "discount_dependency",
        "emotional_purchase_index",
    ]
    scaler = RobustScaler()
    normalized[behavior_columns] = scaler.fit_transform(normalized[behavior_columns])
    return normalized



def cap_outliers(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    capped = dataframe.copy()
    summary: dict[str, Any] = {}
    for column in RAW_NUMERIC_FEATURES:
        lower_quartile = capped[column].quantile(0.25)
        upper_quartile = capped[column].quantile(0.75)
        interquartile_range = upper_quartile - lower_quartile
        lower_bound = lower_quartile - 1.5 * interquartile_range
        upper_bound = upper_quartile + 1.5 * interquartile_range
        summary[column] = {
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
            "clipped_low": int((capped[column] < lower_bound).sum()),
            "clipped_high": int((capped[column] > upper_bound).sum()),
        }
        capped[column] = capped[column].clip(lower_bound, upper_bound)
    return capped, summary



def summarize_behavior_scale(dataframe: pd.DataFrame) -> dict[str, Any]:
    behavior_columns = [
        "impulse_score",
        "spending_efficiency",
        "shopping_intensity",
        "engagement_score",
        "conversion_probability",
        "browsing_pressure",
        "discount_dependency",
        "emotional_purchase_index",
    ]
    summary = {}
    for column in behavior_columns:
        series = dataframe[column]
        summary[column] = {
            "median": round(float(series.median()), 4),
            "iqr": round(float(series.quantile(0.75) - series.quantile(0.25)), 4),
            "p95": round(float(series.quantile(0.95)), 4),
        }
    return summary



def analyze_session_behavior(dataframe: pd.DataFrame) -> dict[str, Any]:
    session_frame = dataframe.copy()
    session_frame["session_depth_estimate"] = session_frame["browsing_time"] * 0.55 + session_frame["cart_items"] * 2.1 + session_frame["impulsive_clicks"] * 0.9
    session_frame["night_session_ratio"] = np.where(session_frame["late_night_activity"] > 50, 1.0, 0.0)
    session_frame["peak_session_score"] = session_frame["session_depth_estimate"] * 0.65 + session_frame["night_session_ratio"] * 8 + session_frame["engagement_score"] * 0.12
    grouped = session_frame.groupby(TARGET).agg(
        browsing_time_mean=("browsing_time", "mean"),
        cart_items_mean=("cart_items", "mean"),
        impulsive_clicks_mean=("impulsive_clicks", "mean"),
        session_depth_mean=("session_depth_estimate", "mean"),
        night_session_rate=("night_session_ratio", "mean"),
        peak_session_score=("peak_session_score", "mean"),
    )
    return grouped.round(4).to_dict(orient="index")



def segment_customers(dataframe: pd.DataFrame) -> dict[str, Any]:
    segmented = dataframe.copy()
    conditions = [
        (segmented["engagement_score"] >= segmented["engagement_score"].quantile(0.78)) & (segmented["conversion_probability"] >= segmented["conversion_probability"].quantile(0.7)),
        (segmented["discount_dependency"] >= segmented["discount_dependency"].quantile(0.75)),
        (segmented["income"] >= segmented["income"].quantile(0.75)) & (segmented["avg_spending"] >= segmented["avg_spending"].quantile(0.72)),
        (segmented["engagement_score"] <= segmented["engagement_score"].quantile(0.25)) & (segmented["cart_items"] <= segmented["cart_items"].quantile(0.35)),
    ]
    choices = ["High Intent", "Discount Driven", "Premium Explorer", "Low Intent"]
    segmented["segment"] = np.select(conditions, choices, default="Balanced Browser")
    return {
        "segment_counts": segmented["segment"].value_counts().to_dict(),
        "segment_means": segmented.groupby("segment")[["engagement_score", "conversion_probability", "discount_dependency", "avg_spending"]].mean().round(4).to_dict(orient="index"),
    }



def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )



def compute_cv_score(estimator: Pipeline, features: pd.DataFrame, target: pd.Series) -> float:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(estimator, features, target, cv=splitter, scoring="accuracy", n_jobs=1)
    return float(scores["test_score"].mean())



def train_models(train_features: pd.DataFrame, train_target: pd.Series, test_features: pd.DataFrame, test_target: pd.Series) -> tuple[dict[str, ModelResult], str, Pipeline]:
    models: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "Logistic Regression": (
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            solver="lbfgs",
                            C=1.8,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.8, 1.2, 1.8]},
        ),
        "Random Forest": (
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=280,
                            max_depth=12,
                            min_samples_split=4,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {"model__max_depth": [10, 12, 15], "model__min_samples_leaf": [1, 2, 3]},
        ),
        "XGBoost": (
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "model",
                        XGBClassifier(
                            objective="multi:softmax",
                            num_class=len(MOODS),
                            n_estimators=320,
                            learning_rate=0.06,
                            max_depth=5,
                            subsample=0.9,
                            colsample_bytree=0.85,
                            reg_lambda=1.2,
                            reg_alpha=0.2,
                            min_child_weight=1.8,
                            tree_method="hist",
                            eval_metric="mlogloss",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__max_depth": [4, 5, 6], "model__learning_rate": [0.05, 0.06, 0.08]},
        ),
    }

    results: dict[str, ModelResult] = {}
    for model_name, (pipeline, params) in models.items():
        search = GridSearchCV(pipeline, params, cv=3, scoring="accuracy", n_jobs=1)
        search.fit(train_features, train_target)
        best_estimator = search.best_estimator_
        predictions = best_estimator.predict(test_features)
        result = ModelResult(
            name=model_name,
            accuracy=float(accuracy_score(test_target, predictions)),
            precision=float(precision_score(test_target, predictions, average="weighted", zero_division=0)),
            recall=float(recall_score(test_target, predictions, average="weighted", zero_division=0)),
            f1=float(f1_score(test_target, predictions, average="weighted", zero_division=0)),
            cv_accuracy=float(search.best_score_),
            estimator=best_estimator,
            predictions=predictions,
        )
        results[model_name] = result

    best_name = max(results, key=lambda name: results[name].accuracy)
    return results, best_name, results[best_name].estimator



def get_transformed_feature_names(estimator: Pipeline) -> list[str]:
    preprocessor: ColumnTransformer = estimator.named_steps["preprocessor"]
    numeric_names = NUMERIC_FEATURES
    categorical_encoder: OneHotEncoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    categorical_names = list(categorical_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return numeric_names + categorical_names



def plot_mood_distribution(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    order = dataframe[TARGET].value_counts().index
    ax = sns.countplot(data=dataframe, x=TARGET, order=order, hue=TARGET, palette="Spectral", legend=False)
    total = float(len(dataframe))
    for patch in ax.patches:
        height = patch.get_height()
        if height:
            ax.annotate(
                f"{int(height)}\n({height / total:.0%})",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.xticks(rotation=20, ha="right")
    plt.title("Buying Mood Distribution")
    plt.ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "mood_distribution.png", dpi=220)
    plt.close()



def plot_spending_behavior(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    ax = sns.boxplot(data=dataframe, x=TARGET, y="avg_spending", hue=TARGET, palette="viridis", legend=False)
    sns.stripplot(data=dataframe.sample(min(len(dataframe), 500), random_state=RANDOM_STATE), x=TARGET, y="avg_spending", color="white", alpha=0.28, size=2, jitter=0.22)
    for mood, value in dataframe.groupby(TARGET)["avg_spending"].mean().items():
        ax.axhline(value, linestyle="--", linewidth=1, alpha=0.18, color="black")
    plt.xticks(rotation=20, ha="right")
    plt.title("Spending Behavior by Buying Mood")
    plt.ylabel("Average Spending ($)")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "spending_behavior.png", dpi=220)
    plt.close()



def plot_category_preferences(dataframe: pd.DataFrame) -> None:
    preferred = dataframe.groupby(["preferred_category", TARGET]).size().reset_index(name="count")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=preferred, x="preferred_category", y="count", hue=TARGET, palette="tab10")
    plt.xticks(rotation=20, ha="right")
    plt.title("Category Preferences Across Buying Moods")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "category_preferences.png", dpi=220)
    plt.close()



def plot_impulse_heatmap(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 9))
    corr_columns = [
        "age",
        "income",
        "browsing_time",
        "cart_items",
        "purchase_frequency",
        "discount_usage",
        "avg_spending",
        "late_night_activity",
        "impulsive_clicks",
        "impulse_score",
        "spending_efficiency",
        "shopping_intensity",
        "browsing_pressure",
        "discount_dependency",
        "emotional_purchase_index",
    ]
    corr = dataframe[corr_columns].corr()
    focus_columns = ["discount_usage", "purchase_frequency", "avg_spending", "income", "discount_dependency"]
    focus = corr.loc[focus_columns, focus_columns]
    sns.heatmap(focus, cmap="coolwarm", center=0, linewidths=0.5, annot=True, fmt=".2f", square=True)
    plt.title("Discount Dependency and Behavioral Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "impulse_heatmap.png", dpi=220)
    plt.close()



def plot_late_night_activity(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    sns.violinplot(data=dataframe, x=TARGET, y="late_night_activity", hue=TARGET, palette="magma", inner="quartile", legend=False)
    plt.xticks(rotation=20, ha="right")
    plt.title("Late Night Shopping Activity")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "late_night_activity.png", dpi=220)
    plt.close()



def plot_discount_usage_analysis(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=dataframe, x=TARGET, y="discount_usage", hue=TARGET, palette="cividis", legend=False)
    plt.xticks(rotation=20, ha="right")
    plt.title("Discount Usage Analysis")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "discount_usage_analysis.png", dpi=220)
    plt.close()



def plot_feature_importance(estimator: Pipeline) -> None:
    model = estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_).mean(axis=0)
    feature_names = get_transformed_feature_names(estimator)
    series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(18)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=series.values, y=series.index, hue=series.index, palette="crest", legend=False)
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "feature_importance.png", dpi=220)
    plt.close()



def plot_confusion_matrix(target: np.ndarray, predictions: np.ndarray) -> None:
    matrix = confusion_matrix(target, predictions, labels=list(range(len(MOODS))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=MOODS, yticklabels=MOODS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "confusion_matrix.png", dpi=220)
    plt.close()



def export_model_outputs(results: dict[str, ModelResult], best_name: str, best_estimator: Pipeline, test_features: pd.DataFrame, test_target: np.ndarray, dataset: pd.DataFrame) -> dict[str, Any]:
    best_predictions = results[best_name].predictions
    joblib.dump(best_estimator, MODELS_DIR / "buying_mood_predictor.pkl")
    plot_confusion_matrix(test_target, best_predictions)
    plot_feature_importance(best_estimator)

    report = classification_report(test_target, best_predictions, labels=list(range(len(MOODS))), target_names=MOODS, digits=4, zero_division=0)
    (METRICS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    metrics_payload = {
        "best_model": best_name,
        "dataset_rows": int(len(dataset)),
        "class_balance": dataset[TARGET].value_counts().to_dict(),
        "models": {
            name: {
                "accuracy": round(result.accuracy, 4),
                "precision": round(result.precision, 4),
                "recall": round(result.recall, 4),
                "f1": round(result.f1, 4),
                "cv_accuracy": round(result.cv_accuracy, 4),
            }
            for name, result in results.items()
        },
    }
    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report_markdown = [
        "# ShopShift AI Project Report",
        "",
        "## Objective",
        "Predict consumer buying mood from shopping behavior and surface behavioral signals tied to impulsive, luxury, budget-conscious, window-shopping, and emotional patterns.",
        "",
        "## Best Model",
        f"- **{best_name}**",
        f"- Test Accuracy: {results[best_name].accuracy:.4f}",
        f"- Weighted F1: {results[best_name].f1:.4f}",
        "",
        "## Model Comparison",
    ]
    for name, result in results.items():
        report_markdown.append(
            f"- {name}: accuracy={result.accuracy:.4f}, precision={result.precision:.4f}, recall={result.recall:.4f}, f1={result.f1:.4f}, cv={result.cv_accuracy:.4f}"
        )
    report_markdown.extend(
        [
            "",
            "## Behavioral Insights",
            "- Late night activity and impulsive clicks separate emotional and impulsive buyers.",
            "- Income and spending are the strongest luxury-oriented signals.",
            "- Discount usage and lower purchase frequency identify budget-conscious behavior.",
            "- Browsing pressure and low cart conversion shape the window shopper class.",
            "",
            "## Outputs",
            "- Visuals exported to `visuals/`",
            "- Model artifact exported to `models/`",
            "- Reports exported to `reports/` and `metrics/`",
        ]
    )
    (REPORTS_DIR / "project_report.md").write_text("\n".join(report_markdown), encoding="utf-8")

    return metrics_payload



def write_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [
                    "# ShopShift AI\n",
                    "Consumer Buying Mood Analytics System\n",
                    "\n",
                    "This notebook runs the complete analytics pipeline, creates behavioral features, trains the mood models, and exports the project artifacts.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "from src.shopshift_ai.pipeline import main\n",
                    "main()\n",
                ],
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [
                    "## Expected Outputs\n",
                    "The execution generates the dataset, charts, trained model, metrics, and project report.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOK_DIR / "shopshift_analysis.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")



def evaluate_and_export() -> dict[str, Any]:
    ensure_directories()
    raw_data = generate_synthetic_dataset()
    raw_data.to_csv(DATA_DIR / "synthetic_shopping_behavior.csv", index=False)
    data_quality = validate_dataset(raw_data)
    DATA_QUALITY_REPORT.write_text(json.dumps(data_quality, indent=2), encoding="utf-8")

    noisy_data = add_missing_values(raw_data)
    missing_profile = summarize_missingness(noisy_data)
    MISSING_VALUE_REPORT.write_text(json.dumps(missing_profile, indent=2), encoding="utf-8")
    cleaned = clean_data(noisy_data)
    imputed = impute_missing_values(cleaned)
    categorical_profile = summarize_categorical_distribution(imputed)
    CATEGORICAL_PROFILE_REPORT.write_text(json.dumps(categorical_profile, indent=2), encoding="utf-8")
    imputed = normalize_categorical_fields(imputed)
    outlier_capped, outlier_profile = cap_outliers(imputed)
    OUTLIER_REPORT.write_text(json.dumps(outlier_profile, indent=2), encoding="utf-8")
    engineered = engineer_features(outlier_capped)
    normalized_behaviors = normalize_behavioral_metrics(engineered)
    behavior_profile = {
        column: {
            "mean": round(float(normalized_behaviors[column].mean()), 4),
            "std": round(float(normalized_behaviors[column].std(ddof=0)), 4),
        }
        for column in [
            "impulse_score",
            "spending_efficiency",
            "shopping_intensity",
            "engagement_score",
            "conversion_probability",
            "browsing_pressure",
            "discount_dependency",
            "emotional_purchase_index",
        ]
    }
    NORMALIZED_BEHAVIOR_REPORT.write_text(json.dumps(behavior_profile, indent=2), encoding="utf-8")
    BEHAVIOR_SCALE_REPORT.write_text(json.dumps(summarize_behavior_scale(engineered), indent=2), encoding="utf-8")
    session_report = analyze_session_behavior(engineered)
    SESSION_ANALYSIS_REPORT.write_text(json.dumps(session_report, indent=2), encoding="utf-8")
    segment_report = segment_customers(engineered)
    CUSTOMER_SEGMENT_REPORT.write_text(json.dumps(segment_report, indent=2), encoding="utf-8")
    normalized_preview = normalize_numeric_columns(engineered)
    _ = normalized_preview

    features = engineered.drop(columns=["user_id", TARGET])
    target = engineered[TARGET]
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        encoded_target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=encoded_target,
    )

    results, best_name, best_estimator = train_models(train_features, train_target, test_features, test_target)

    plot_mood_distribution(engineered)
    plot_spending_behavior(engineered)
    plot_category_preferences(engineered)
    plot_impulse_heatmap(engineered)
    plot_late_night_activity(engineered)
    plot_discount_usage_analysis(engineered)

    metrics_payload = export_model_outputs(results, best_name, best_estimator, test_features, test_target, engineered)
    write_notebook()
    return metrics_payload



def main() -> None:
    metrics = evaluate_and_export()
    summary = {
        "best_model": metrics["best_model"],
        "models": metrics["models"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
