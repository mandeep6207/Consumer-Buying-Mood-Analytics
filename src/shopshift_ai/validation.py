from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass
class DatasetQualitySummary:
    row_count: int
    column_count: int
    duplicate_users: int
    missing_values: dict[str, int]
    class_balance: dict[str, int]
    numeric_ranges: dict[str, dict[str, float]]


def validate_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    required_columns = {
        "user_id",
        "age",
        "gender",
        "income",
        "browsing_time",
        "cart_items",
        "purchase_frequency",
        "discount_usage",
        "avg_spending",
        "preferred_category",
        "late_night_activity",
        "impulsive_clicks",
        "buying_mood",
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    numeric_columns = [
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
    quality = DatasetQualitySummary(
        row_count=int(len(dataframe)),
        column_count=int(dataframe.shape[1]),
        duplicate_users=int(dataframe["user_id"].duplicated().sum()) if "user_id" in dataframe else 0,
        missing_values=dataframe.isna().sum().to_dict(),
        class_balance=dataframe["buying_mood"].value_counts().to_dict() if "buying_mood" in dataframe else {},
        numeric_ranges={
            column: {
                "min": float(pd.to_numeric(dataframe[column], errors="coerce").min()),
                "max": float(pd.to_numeric(dataframe[column], errors="coerce").max()),
            }
            for column in numeric_columns
            if column in dataframe
        },
    )
    return {
        "missing_columns": missing_columns,
        "summary": asdict(quality),
    }