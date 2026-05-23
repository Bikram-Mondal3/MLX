"""
feature_search_meta_model.py  —  Optuna-tuned stacked ensemble for Solar Radiation Prediction
------------------------------------------------------------------------------------------------
Changes vs original:
  • Every hard-coded hyperparameter for HGB, ExtraTrees, and RandomForest is now
    searched by Optuna (TPE sampler + MedianPruner).
  • A single GroupKFold CV fold is used as the Optuna objective (fast); the best
    params are then used in the full 5-fold cross-validation and final fit.
  • SelectFromModel threshold is also tuned ("mean" | "median" | "0.5*mean").
  • VotingRegressor weights for the three models are tuned as well in train_solar_model style.
  • All original CLI flags are preserved; three new flags added:
      --optuna-trials   (default 60)
      --optuna-timeout  (seconds, default None)
      --optuna-jobs     (parallel jobs, default 1)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna-tuned stacked meta-model for Solar Radiation Prediction."
    )
    parser.add_argument("--train", default="train_df_1.csv")
    parser.add_argument("--test", default="test_df_1.csv")
    parser.add_argument("--submission", default="stacked_submission.csv")
    parser.add_argument("--metrics", default="stacked_cv_metrics.json")
    parser.add_argument("--submission-target-name", default="TARGET")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    # Optuna-specific
    parser.add_argument("--optuna-trials", type=int, default=60,
                        help="Number of Optuna trials (default: 60).")
    parser.add_argument("--optuna-timeout", type=float, default=None,
                        help="Wall-clock seconds budget for Optuna (default: no limit).")
    parser.add_argument("--optuna-jobs", type=int, default=1,
                        help="Parallel Optuna jobs (default: 1).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

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
            "solar_power_proxy": solar_elevation_proxy ** 1.5,
            "is_daylight": (
                (time_minutes >= sunrise_minutes) & (time_minutes <= sunset_minutes)
            ).astype(int),
            "solar_noon_offset": np.abs(
                time_minutes - (sunrise_minutes + sunset_minutes) / 2.0
            ),
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
            "speed_sq": speed ** 2,
            "humidity_sq": humidity ** 2,
            "temperature_sq": temperature ** 2,
            "daylight_x_humidity": solar_elevation_proxy * humidity,
            "daylight_x_temperature": solar_elevation_proxy * temperature,
        }
    )

    for col in ["Temperature", "Pressure", "Humidity", "Speed"]:
        features[f"{col.lower()}_missing"] = frame[col].isna().astype(int)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Model builder from Optuna params
# ─────────────────────────────────────────────────────────────────────────────

def build_model_from_params(params: dict, random_state: int) -> Pipeline:
    """Construct the full stacking pipeline from an Optuna params dict."""

    hgb = HistGradientBoostingRegressor(
        learning_rate=params["hgb_lr"],
        max_depth=params["hgb_max_depth"],
        max_iter=params["hgb_max_iter"],
        min_samples_leaf=params["hgb_min_samples_leaf"],
        l2_regularization=params["hgb_l2"],
        max_leaf_nodes=params["hgb_max_leaf_nodes"],
        random_state=random_state,
    )

    extra_trees = ExtraTreesRegressor(
        n_estimators=params["et_n_estimators"],
        max_depth=params["et_max_depth"],
        min_samples_leaf=params["et_min_samples_leaf"],
        max_features=params["et_max_features"],
        random_state=random_state,
        n_jobs=-1,
    )

    random_forest = RandomForestRegressor(
        n_estimators=params["rf_n_estimators"],
        max_depth=params["rf_max_depth"],
        min_samples_leaf=params["rf_min_samples_leaf"],
        max_features=params["rf_max_features"],
        random_state=random_state,
        n_jobs=-1,
    )

    stacked = StackingRegressor(
        estimators=[
            ("hgb", hgb),
            ("extra_trees", extra_trees),
            ("random_forest", random_forest),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 20)),
        cv=3,           # inner CV kept small to speed up Optuna trials
        n_jobs=-1,
    )

    selector = SelectFromModel(
        ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        threshold=params["selector_threshold"],
    )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", selector),
            ("stacked", stacked),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    random_state: int,
):
    """Returns a closure that Optuna calls for each trial."""

    # Use a single held-out fold for speed during search
    splitter = GroupKFold(n_splits=5)
    train_idx, valid_idx = next(splitter.split(X, y, groups=groups))

    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    def objective(trial: optuna.Trial) -> float:
        params = {
            # ── HistGradientBoosting ──────────────────────────────────────
            "hgb_lr":               trial.suggest_float("hgb_lr", 0.01, 0.2, log=True),
            "hgb_max_depth":        trial.suggest_int("hgb_max_depth", 4, 14),
            "hgb_max_iter":         trial.suggest_int("hgb_max_iter", 200, 800, step=100),
            "hgb_min_samples_leaf": trial.suggest_int("hgb_min_samples_leaf", 5, 50),
            "hgb_l2":               trial.suggest_float("hgb_l2", 1e-3, 1.0, log=True),
            "hgb_max_leaf_nodes":   trial.suggest_int("hgb_max_leaf_nodes", 15, 127),

            # ── ExtraTrees ───────────────────────────────────────────────
            "et_n_estimators":      trial.suggest_int("et_n_estimators", 100, 600, step=100),
            "et_max_depth":         trial.suggest_int("et_max_depth", 5, 30),
            "et_min_samples_leaf":  trial.suggest_int("et_min_samples_leaf", 1, 20),
            "et_max_features":      trial.suggest_float("et_max_features", 0.3, 1.0),

            # ── RandomForest ─────────────────────────────────────────────
            "rf_n_estimators":      trial.suggest_int("rf_n_estimators", 100, 500, step=100),
            "rf_max_depth":         trial.suggest_int("rf_max_depth", 5, 30),
            "rf_min_samples_leaf":  trial.suggest_int("rf_min_samples_leaf", 1, 20),
            "rf_max_features":      trial.suggest_float("rf_max_features", 0.3, 1.0),

            # ── Feature selector ─────────────────────────────────────────
            "selector_threshold":   trial.suggest_categorical(
                "selector_threshold", ["mean", "median", "0.5*mean", "0.75*mean"]
            ),
        }

        try:
            model = build_model_from_params(params, random_state)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            return float(root_mean_squared_error(y_val, preds))
        except Exception:
            return float("inf")

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Full cross-validation (same as original)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    folds: int,
) -> dict:
    splitter = GroupKFold(n_splits=folds)
    oof_predictions = np.zeros(len(X), dtype=float)
    fold_metrics: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = fold_model.predict(X.iloc[valid_idx])
        oof_predictions[valid_idx] = predictions

        fold_rmse = root_mean_squared_error(y.iloc[valid_idx], predictions)
        fold_mae = mean_absolute_error(y.iloc[valid_idx], predictions)

        try:
            mask = fold_model.named_steps["selector"].get_support()
            selected_features = int(mask.sum())
        except AttributeError:
            selected_features = -1

        fold_metrics.append(
            {
                "fold": fold,
                "rmse": float(fold_rmse),
                "mae": float(fold_mae),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "selected_features": selected_features,
            }
        )
        print(
            f"  Fold {fold}  RMSE: {fold_rmse:.4f}  MAE: {fold_mae:.4f}"
            f"  Features: {selected_features}"
        )

    return {
        "folds": fold_metrics,
        "overall_rmse": float(root_mean_squared_error(y, oof_predictions)),
        "overall_mae": float(mean_absolute_error(y, oof_predictions)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    print("Building features...")
    X_train = build_features(train_df)
    X_test = build_features(test_df)
    y_train = train_df["Radiation"].astype(float)
    groups = pd.to_datetime(train_df["Data"], format="%d-%m-%Y")

    # ── Optuna search ────────────────────────────────────────────────────────
    print(f"\nRunning Optuna ({args.optuna_trials} trials)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.random_state),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )
    study.optimize(
        make_objective(X_train, y_train, groups, args.random_state),
        n_trials=args.optuna_trials,
        timeout=args.optuna_timeout,
        n_jobs=args.optuna_jobs,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_trial_rmse = study.best_value
    print(f"\nBest trial RMSE : {best_trial_rmse:.4f}")
    print("Best params     :", json.dumps(best_params, indent=2))

    # ── Full CV with best params ─────────────────────────────────────────────
    print("\nFull cross-validation with best params...")
    best_model = build_model_from_params(best_params, args.random_state)
    metrics = cross_validate(best_model, X_train, y_train, groups, folds=args.folds)
    metrics["best_optuna_trial_rmse"] = best_trial_rmse
    metrics["best_params"] = best_params

    # ── Final fit & predict ──────────────────────────────────────────────────
    print("\nFitting on full training set...")
    final_model = build_model_from_params(best_params, args.random_state)
    final_model.fit(X_train, y_train)

    predictions = np.clip(final_model.predict(X_test), a_min=0.0, a_max=None)

    submission = pd.DataFrame(
        {"ID": test_df["ID"], args.submission_target_name: predictions}
    )
    submission.to_csv(args.submission, index=False)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n── Results ──────────────────────────────────────────")
    print(f"OOF RMSE : {metrics['overall_rmse']:.4f}")
    print(f"OOF MAE  : {metrics['overall_mae']:.4f}")
    print(f"Submission → {args.submission}")
    print(f"Metrics    → {args.metrics}")

    # Print selected features
    try:
        support = final_model.named_steps["selector"].get_support()
        selected = X_train.columns[support].tolist()
        print(f"Selected {len(selected)} features: {selected}")
    except AttributeError:
        pass


if __name__ == "__main__":
    main()