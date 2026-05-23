from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a solar radiation regression ensemble and create a submission file.")
    parser.add_argument("--train", default="train_df_1.csv", help="Path to training CSV.")
    parser.add_argument("--test", default="test_df_1.csv", help="Path to test CSV.")
    parser.add_argument("--submission", default="submission.csv", help="Output submission CSV path.")
    parser.add_argument("--metrics", default="cv_metrics.json", help="Output CV metrics JSON path.")
    parser.add_argument("--submission-target-name", default="TARGET", help="Prediction column name for the submission file.")
    parser.add_argument("--folds", type=int, default=5, help="Number of GroupKFold splits.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    date = pd.to_datetime(frame["Data"], format="%d-%m-%Y")
    time = pd.to_timedelta(frame["Time"])
    sunrise = pd.to_timedelta(frame["TimeSunRise"])
    sunset = pd.to_timedelta(frame["TimeSunSet"])
    timestamp = date + time

    time_minutes = time.dt.total_seconds() / 60.0
    sunrise_minutes = sunrise.dt.total_seconds() / 60.0
    sunset_minutes = sunset.dt.total_seconds() / 60.0
    daylight_minutes = sunset_minutes - sunrise_minutes
    since_sunrise = time_minutes - sunrise_minutes
    until_sunset = sunset_minutes - time_minutes
    daylight_progress = np.clip(since_sunrise / daylight_minutes.replace(0, np.nan), 0.0, 1.0)
    solar_arc = np.pi * daylight_progress
    solar_elevation_proxy = np.sin(solar_arc).clip(lower=0.0)

    temperature = pd.to_numeric(frame["Temperature"], errors="coerce")
    pressure = pd.to_numeric(frame["Pressure"], errors="coerce")
    humidity = pd.to_numeric(frame["Humidity"], errors="coerce")
    speed = pd.to_numeric(frame["Speed"], errors="coerce")
    wind_direction = pd.to_numeric(frame["WindDirection(Degrees)"], errors="coerce")

    features = pd.DataFrame(
        {
            "unix_time": pd.to_numeric(frame["UNIXTime"], errors="coerce"),
            "temperature": temperature,
            "pressure": pressure,
            "humidity": humidity,
            "speed": speed,
            "wind_direction": wind_direction,
            "month": timestamp.dt.month,
            "day": timestamp.dt.day,
            "dayofweek": timestamp.dt.dayofweek,
            "dayofyear": timestamp.dt.dayofyear,
            "weekofyear": timestamp.dt.isocalendar().week.astype(int),
            "hour": timestamp.dt.hour,
            "minute": timestamp.dt.minute,
            "second": timestamp.dt.second,
            "time_minutes": time_minutes,
            "sunrise_minutes": sunrise_minutes,
            "sunset_minutes": sunset_minutes,
            "daylight_minutes": daylight_minutes,
            "since_sunrise": since_sunrise,
            "until_sunset": until_sunset,
            "daylight_progress": daylight_progress,
            "solar_elevation_proxy": solar_elevation_proxy,
            "solar_power_proxy": solar_elevation_proxy**1.5,
            "is_daylight": ((time_minutes >= sunrise_minutes) & (time_minutes <= sunset_minutes)).astype(int),
            "solar_noon_offset": np.abs(time_minutes - (sunrise_minutes + sunset_minutes) / 2.0),
            "wind_dir_sin": np.sin(np.deg2rad(wind_direction)),
            "wind_dir_cos": np.cos(np.deg2rad(wind_direction)),
            "hour_sin": np.sin(2 * np.pi * timestamp.dt.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * timestamp.dt.hour / 24.0),
            "dayofyear_sin": np.sin(2 * np.pi * timestamp.dt.dayofyear / 366.0),
            "dayofyear_cos": np.cos(2 * np.pi * timestamp.dt.dayofyear / 366.0),
            "temp_x_humidity": temperature * humidity,
            "temp_x_pressure": temperature * pressure,
            "wind_x_humidity": speed * humidity,
            "wind_x_temp": speed * temperature,
            "pressure_x_humidity": pressure * humidity,
            "speed_sq": speed**2,
            "humidity_sq": humidity**2,
            "temperature_sq": temperature**2,
            "daylight_x_humidity": solar_elevation_proxy * humidity,
            "daylight_x_temperature": solar_elevation_proxy * temperature,
        }
    )

    for col in ["Temperature", "Pressure", "Humidity", "Speed"]:
        features[f"{col.lower()}_missing"] = frame[col].isna().astype(int)

    return features


def build_ensemble(random_state: int) -> VotingRegressor:
    hgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=8,
                    max_iter=500,
                    min_samples_leaf=20,
                    l2_regularization=0.10,
                    random_state=random_state,
                ),
            ),
        ]
    )

    extra_trees = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return VotingRegressor(
        estimators=[
            ("hgb", hgb),
            ("extra_trees", extra_trees),
            ("random_forest", random_forest),
        ],
        weights=[0.50, 0.35, 0.15],
        n_jobs=-1,
    )


def cross_validate(model: VotingRegressor, X: pd.DataFrame, y: pd.Series, groups: pd.Series, folds: int) -> dict:
    splitter = GroupKFold(n_splits=folds)
    oof_predictions = np.zeros(len(X), dtype=float)
    fold_metrics: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = fold_model.predict(X.iloc[valid_idx])
        oof_predictions[valid_idx] = predictions

        fold_rmse = root_mean_squared_error(y.iloc[valid_idx], predictions)
        fold_mae = mean_absolute_error(y.iloc[valid_idx], predictions)
        fold_metrics.append(
            {
                "fold": fold,
                "rmse": float(fold_rmse),
                "mae": float(fold_mae),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )

    metrics = {
        "folds": fold_metrics,
        "overall_rmse": float(root_mean_squared_error(y, oof_predictions)),
        "overall_mae": float(mean_absolute_error(y, oof_predictions)),
    }
    return metrics


def main() -> None:
    args = parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    submission_path = Path(args.submission)
    metrics_path = Path(args.metrics)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = build_features(train_df)
    X_test = build_features(test_df)
    y_train = train_df["Radiation"].astype(float)
    groups = pd.to_datetime(train_df["Data"], format="%d-%m-%Y")

    model = build_ensemble(random_state=args.random_state)
    metrics = cross_validate(model, X_train, y_train, groups, folds=args.folds)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    submission = pd.DataFrame(
        {
            "ID": test_df["ID"],
            args.submission_target_name: predictions,
        }
    )

    submission.to_csv(submission_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved submission to {submission_path}")
    print(f"Saved CV metrics to {metrics_path}")
    print(f"OOF RMSE: {metrics['overall_rmse']:.4f}")
    print(f"OOF MAE: {metrics['overall_mae']:.4f}")


if __name__ == "__main__":
    main()
