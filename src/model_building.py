import json
import os
from pathlib import Path
import shutil
import statistics
import time
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import yaml

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAMS_PATH = Path("params.yaml")
X_TRAIN_PATH = Path("data/processed/X_train.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")

MODELS_DIR = Path("models")
CHAMPION_MODEL_PATH = MODELS_DIR / "model.pkl"
SELECTION_REPORT_PATH = Path("reports/model_selection.json")
COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
COMPARISON_MD_PATH = Path("reports/model_comparison.md")
REQUIREMENTS_OUT_PATH = Path("requirements.txt")

NUMERIC_COLUMNS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]

CLASSIFICATION_MODELS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "GaussianNB",
    "SVM",
    "KNN",
]

REGRESSION_MODELS = [
    "Ridge",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "SVM",
    "KNN",
]


def load_params() -> dict:
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_problem_type(y: pd.Series) -> str:
    """Automatically detect whether target represents Classification or Regression."""
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    unique_count = y.nunique()
    if unique_count <= 20 and np.all(np.equal(np.mod(y.dropna(), 1), 0)):
        return "classification"
    return "regression"


def build_preprocessor(scaler_type: str) -> ColumnTransformer:
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", scaler, NUMERIC_COLUMNS),
            ("cat", ohe, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def suggest_params(trial: optuna.Trial, model_name: str, problem_type: str, random_state: int) -> dict:
    params = {
        "scaler_type": trial.suggest_categorical("scaler_type", ["standard", "minmax"]),
    }

    if problem_type == "classification":
        if model_name == "LogisticRegression":
            params["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
            params["max_iter"] = 500
            params["random_state"] = random_state
        elif model_name == "DecisionTree":
            params["max_depth"] = trial.suggest_int("max_depth", 3, 18)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
            params["random_state"] = random_state
        elif model_name == "RandomForest":
            params["n_estimators"] = trial.suggest_int("n_estimators", 40, 120, step=20)
            params["max_depth"] = trial.suggest_int("max_depth", 4, 16)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 16)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 8)
            params["random_state"] = random_state
            params["n_jobs"] = -1
        elif model_name == "GradientBoosting":
            params["max_iter"] = trial.suggest_int("max_iter", 50, 150, step=25)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 10, 50)
            params["random_state"] = random_state
        elif model_name == "GaussianNB":
            params["var_smoothing"] = trial.suggest_float("var_smoothing", 1e-11, 1e-6, log=True)
        elif model_name == "SVM":
            params["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
            params["random_state"] = random_state
            params["max_iter"] = 2000
        elif model_name == "KNN":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 21, step=2)
            params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
            params["p"] = trial.suggest_int("p", 1, 2)
            params["n_jobs"] = -1
    else:
        if model_name == "Ridge":
            params["alpha"] = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
            params["random_state"] = random_state
        elif model_name == "DecisionTree":
            params["max_depth"] = trial.suggest_int("max_depth", 3, 18)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
            params["random_state"] = random_state
        elif model_name == "RandomForest":
            params["n_estimators"] = trial.suggest_int("n_estimators", 40, 120, step=20)
            params["max_depth"] = trial.suggest_int("max_depth", 4, 16)
            params["random_state"] = random_state
            params["n_jobs"] = -1
        elif model_name == "GradientBoosting":
            params["max_iter"] = trial.suggest_int("max_iter", 50, 150, step=25)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["random_state"] = random_state
        elif model_name == "SVM":
            params["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
            params["random_state"] = random_state
        elif model_name == "KNN":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 21, step=2)
            params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
            params["n_jobs"] = -1

    return params


def build_pipeline(model_name: str, params: dict, problem_type: str) -> Pipeline:
    model_kwargs = {k: v for k, v in params.items() if k != "scaler_type"}
    preprocessor = build_preprocessor(params["scaler_type"])

    if problem_type == "classification":
        if model_name == "LogisticRegression":
            estimator = LogisticRegression(**model_kwargs)
        elif model_name == "DecisionTree":
            estimator = DecisionTreeClassifier(**model_kwargs)
        elif model_name == "RandomForest":
            estimator = RandomForestClassifier(**model_kwargs)
        elif model_name == "GradientBoosting":
            estimator = HistGradientBoostingClassifier(**model_kwargs)
        elif model_name == "GaussianNB":
            estimator = GaussianNB(**model_kwargs)
        elif model_name == "SVM":
            estimator = LinearSVC(**model_kwargs)
        elif model_name == "KNN":
            estimator = KNeighborsClassifier(**model_kwargs)
        else:
            raise ValueError(f"Unsupported classification model: {model_name}")
    else:
        if model_name == "Ridge":
            estimator = Ridge(**model_kwargs)
        elif model_name == "DecisionTree":
            estimator = DecisionTreeRegressor(**model_kwargs)
        elif model_name == "RandomForest":
            estimator = RandomForestRegressor(**model_kwargs)
        elif model_name == "GradientBoosting":
            estimator = HistGradientBoostingRegressor(**model_kwargs)
        elif model_name == "SVM":
            estimator = LinearSVR(**model_kwargs)
        elif model_name == "KNN":
            estimator = KNeighborsRegressor(**model_kwargs)
        else:
            raise ValueError(f"Unsupported regression model: {model_name}")

    return Pipeline([("Preprocessor", preprocessor), ("Model", estimator)])


def fit_with_timing(
    model_name: str,
    best_params: dict,
    problem_type: str,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    repeats: int,
) -> tuple[Pipeline, float]:
    times = []
    fitted_pipeline = None
    for _ in range(repeats):
        pipeline = build_pipeline(model_name, best_params, problem_type)
        start = time.perf_counter()
        pipeline.fit(X_fit, y_fit)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        fitted_pipeline = pipeline
    return fitted_pipeline, float(statistics.median(times))


def measure_prediction_ms(pipeline: Pipeline, X_sample: pd.DataFrame, n_samples: int, repeats: int) -> float:
    sample_df = X_sample.iloc[: min(n_samples, len(X_sample))]
    for i in range(min(3, len(sample_df))):
        _ = pipeline.predict(sample_df.iloc[[i]])

    per_row_times_ms = []
    for _ in range(repeats):
        for i in range(len(sample_df)):
            row = sample_df.iloc[[i]]
            t0 = time.perf_counter()
            _ = pipeline.predict(row)
            per_row_times_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(per_row_times_ms))


def min_max_normalize(values: list[float]) -> list[float]:
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def select_champion(candidates: list[dict], selection_cfg: dict) -> tuple[dict, list[dict]]:
    max_gap = float(selection_cfg["max_accuracy_gap_pct"])
    tol = float(selection_cfg["accuracy_tolerance_pct"])
    weights = selection_cfg["weights"]
    w_size = float(weights["model_size_mb"])
    w_train = float(weights["training_time_seconds"])
    w_pred = float(weights["prediction_time_ms"])

    for c in candidates:
        c["eligible"] = abs(c["accuracy_gap_pct"]) < max_gap and c["accuracy_gap_pct"] < max_gap

    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        fallback = min(candidates, key=lambda x: abs(x["accuracy_gap_pct"]))
        fallback["eligible"] = True
        eligible = [fallback]

    best_val_acc = max(c["validation_accuracy_pct"] for c in eligible)
    for c in candidates:
        c["finalist"] = c["eligible"] and (best_val_acc - c["validation_accuracy_pct"]) <= tol

    finalists = [c for c in candidates if c["finalist"]]

    norm_sizes = min_max_normalize([c["model_size_mb"] for c in finalists])
    norm_trains = min_max_normalize([c["training_time_seconds"] for c in finalists])
    norm_preds = min_max_normalize([c["prediction_time_ms"] for c in finalists])

    for c in candidates:
        c["norm_size"] = None
        c["norm_train_time"] = None
        c["norm_pred_time"] = None
        c["selection_cost"] = None

    for idx, c in enumerate(finalists):
        c["norm_size"] = round(norm_sizes[idx], 6)
        c["norm_train_time"] = round(norm_trains[idx], 6)
        c["norm_pred_time"] = round(norm_preds[idx], 6)
        c["selection_cost"] = round(
            w_size * norm_sizes[idx] + w_train * norm_trains[idx] + w_pred * norm_preds[idx],
            6,
        )

    finalists_sorted = sorted(
        finalists,
        key=lambda x: (x["selection_cost"], -x["validation_accuracy_pct"], x["model_name"]),
    )
    champion = finalists_sorted[0]
    return champion, candidates


def export_requirements_from_mlflow(run_id: str) -> None:
    """Extract requirements.txt generated by MLflow for the champion model and ensure streamlit is present."""
    try:
        artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/requirements.txt")
        lines = Path(artifact_path).read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = [
            f"mlflow=={mlflow.__version__}",
            "cloudpickle",
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
            "scikit-learn",
            "scipy",
        ]

    if not any(line.strip().lower().startswith("streamlit") for line in lines):
        lines.append("streamlit")

    REQUIREMENTS_OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_markdown(
    problem_type: str,
    champion: dict,
    candidates: list[dict],
    selection_cfg: dict,
    total_train_rows: int,
    tuning_rows: int,
) -> None:
    lines = [
        "# Multi-Criteria Model Comparison Report",
        "",
        f"- **Detected Problem Type:** `{problem_type.upper()}`",
        f"- **Full Training Dataset Rows:** `{total_train_rows:,}`",
        f"- **Stratified Optuna CV Sample Size:** `{tuning_rows:,}`",
        f"- **Selected Champion Model:** **`{champion['model_name']}`** (`models/model.pkl`)",
        f"- **Champion Validation Accuracy:** `{champion['validation_accuracy_pct']:.2f}%`",
        f"- **Champion Selection Cost:** `{champion['selection_cost']:.4f}`",
        "",
        "## Selection Criteria (`params.yaml`)",
        f"- **Max Train-Val Accuracy Gap:** `< {selection_cfg['max_accuracy_gap_pct']}%`",
        f"- **Finalist Accuracy Tolerance Band:** `within {selection_cfg['accuracy_tolerance_pct']}%` of best eligible model",
        f"- **Cost Weights:** Model Size (`{selection_cfg['weights']['model_size_mb']}`), "
        f"Training Time (`{selection_cfg['weights']['training_time_seconds']}`), "
        f"Prediction Latency (`{selection_cfg['weights']['prediction_time_ms']}`)",
        "",
        "## Candidate Models Summary",
        "",
        "| Model | Val Acc (%) | Train Acc (%) | Gap (pp) | Size (MB) | Train Time (s) | Latency (ms) | Eligible | Finalist | Cost |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c in sorted(candidates, key=lambda x: -x["validation_accuracy_pct"]):
        cost_str = f"{c['selection_cost']:.4f}" if c["selection_cost"] is not None else "N/A"
        mark = " **(Champion)**" if c["model_name"] == champion["model_name"] else ""
        lines.append(
            f"| {c['model_name']}{mark} | {c['validation_accuracy_pct']:.2f} | "
            f"{c['training_accuracy_pct']:.2f} | {c['accuracy_gap_pct']:+.2f} | "
            f"{c['model_size_mb']:.4f} | {c['training_time_seconds']:.4f} | "
            f"{c['prediction_time_ms']:.4f} | {'Yes' if c['eligible'] else 'No'} | "
            f"{'Yes' if c['finalist'] else 'No'} | {cost_str} |"
        )

    COMPARISON_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = load_params()
    random_state = int(config["random_state"])
    cv_folds = int(config["cv_folds"])
    n_trials = int(config["n_trials"])
    tuning_sample_size = int(config.get("tuning_sample_size", 25000))
    experiment_name = str(config["experiment_name"])
    selection_cfg = config["selection"]
    measurement_cfg = config["measurement"]

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]

    # Step 1: Detect Problem Type (Classification vs Regression)
    problem_type = detect_problem_type(y_train)
    print(f"[Model Building] Detected Problem Type: {problem_type.upper()} | Training Rows: {len(X_train):,}")

    candidate_names = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
    scoring_metric = "accuracy" if problem_type == "classification" else "r2"

    # Use stratified sample for Optuna 5-fold CV so 7 models x n_trials x 5 folds runs fast
    if len(X_train) > tuning_sample_size:
        X_tune, _, y_tune, _ = train_test_split(
            X_train,
            y_train,
            train_size=tuning_sample_size,
            random_state=random_state,
            stratify=y_train if problem_type == "classification" else None,
        )
        X_tune = X_tune.reset_index(drop=True)
        y_tune = y_tune.reset_index(drop=True)
    else:
        X_tune, y_tune = X_train, y_train

    cv = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        if problem_type == "classification"
        else KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(experiment_name)

    gap_limit = float(selection_cfg["max_accuracy_gap_pct"])
    use_penalty = bool(selection_cfg.get("gap_penalty_in_search", True))
    penalty_weight = float(selection_cfg.get("gap_penalty_weight", 1.0))

    candidates_summary = []

    for model_name in candidate_names:
        print(f"  -> Tuning {model_name} ({n_trials} Optuna trials, {cv_folds}-fold CV)...")
        with mlflow.start_run(run_name=model_name) as parent_run:
            mlflow.set_tag("problem_type", problem_type)
            mlflow.set_tag("model_name", model_name)

            trial_records = {}

            def objective(trial: optuna.Trial) -> float:
                trial_params = suggest_params(trial, model_name, problem_type, random_state)
                pipeline = build_pipeline(model_name, trial_params, problem_type)

                cv_results = cross_validate(
                    pipeline,
                    X_tune,
                    y_tune,
                    cv=cv,
                    scoring=scoring_metric,
                    return_train_score=True,
                    n_jobs=-1,
                )
                val_score = float(np.mean(cv_results["test_score"]))
                train_score = float(np.mean(cv_results["train_score"]))
                gap_pct = (train_score - val_score) * 100.0

                trial_records[trial.number] = {
                    "val_score": val_score,
                    "train_score": train_score,
                    "gap_pct": gap_pct,
                    "params": trial_params,
                }

                with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
                    mlflow.log_params(trial_params)
                    mlflow.log_metrics(
                        {
                            "cv_val_score": val_score,
                            "cv_train_score": train_score,
                            "cv_gap_pct": gap_pct,
                        }
                    )

                obj_val = val_score
                if use_penalty and gap_pct >= gap_limit:
                    obj_val -= penalty_weight * (gap_pct - gap_limit + 1.0) / 100.0
                return obj_val

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=random_state),
            )
            study.optimize(objective, n_trials=n_trials)

            best_trial_num = study.best_trial.number
            best_info = trial_records[best_trial_num]
            best_params = best_info["params"]

            # For KNN (memory-based O(N) distance computation), refit on X_tune to stay responsive;
            # all parametric/tree models refit on the full training dataset X_train.
            X_refit = X_tune if model_name == "KNN" else X_train
            y_refit = y_tune if model_name == "KNN" else y_train

            fitted_pipeline, median_train_time = fit_with_timing(
                model_name=model_name,
                best_params=best_params,
                problem_type=problem_type,
                X_fit=X_refit,
                y_fit=y_refit,
                repeats=int(measurement_cfg["training_time_repeats"]),
            )

            model_path = MODELS_DIR / f"{model_name}.pkl"
            joblib.dump(fitted_pipeline, model_path)
            model_size_mb = model_path.stat().st_size / (1024.0 * 1024.0)

            train_acc_pct = float(fitted_pipeline.score(X_tune, y_tune)) * 100.0
            val_acc_pct = float(best_info["val_score"]) * 100.0
            gap_pct = train_acc_pct - val_acc_pct

            pred_latency_ms = measure_prediction_ms(
                pipeline=fitted_pipeline,
                X_sample=X_tune,
                n_samples=int(measurement_cfg["prediction_n_samples"]),
                repeats=int(measurement_cfg["prediction_repeats"]),
            )

            mlflow.log_params(best_params)
            mlflow.log_metrics(
                {
                    "training_accuracy_pct": train_acc_pct,
                    "validation_accuracy_pct": val_acc_pct,
                    "accuracy_gap_pct": gap_pct,
                    "model_size_mb": model_size_mb,
                    "training_time_seconds": median_train_time,
                    "prediction_time_ms": pred_latency_ms,
                }
            )

            mlflow.sklearn.log_model(
                sk_model=fitted_pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            candidates_summary.append(
                {
                    "model_name": model_name,
                    "run_id": parent_run.info.run_id,
                    "problem_type": problem_type,
                    "best_params": best_params,
                    "training_accuracy_pct": round(train_acc_pct, 4),
                    "validation_accuracy_pct": round(val_acc_pct, 4),
                    "accuracy_gap_pct": round(gap_pct, 4),
                    "model_size_mb": round(model_size_mb, 6),
                    "training_time_seconds": round(median_train_time, 6),
                    "prediction_time_ms": round(pred_latency_ms, 6),
                }
            )
            print(
                f"     {model_name}: Val={val_acc_pct:.2f}%, Train={train_acc_pct:.2f}%, "
                f"Gap={gap_pct:+.2f}pp, Size={model_size_mb:.4f}MB, "
                f"Fit={median_train_time:.3f}s, Pred={pred_latency_ms:.3f}ms"
            )

    champion, updated_candidates = select_champion(candidates_summary, selection_cfg)

    # Copy champion model to models/model.pkl
    shutil.copyfile(MODELS_DIR / f"{champion['model_name']}.pkl", CHAMPION_MODEL_PATH)

    # Export requirements.txt from the winning MLflow model run
    export_requirements_from_mlflow(champion["run_id"])

    selection_payload = {
        "problem_type": problem_type,
        "experiment_name": experiment_name,
        "total_training_rows": len(X_train),
        "tuning_sample_rows": len(X_tune),
        "champion_model": champion["model_name"],
        "champion_run_id": champion["run_id"],
        "champion_metrics": champion,
        "selection_criteria": selection_cfg,
        "all_candidates": updated_candidates,
    }

    SELECTION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=4)

    comparison_df = pd.DataFrame(updated_candidates).drop(columns=["best_params"])
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)

    write_comparison_markdown(
        problem_type=problem_type,
        champion=champion,
        candidates=updated_candidates,
        selection_cfg=selection_cfg,
        total_train_rows=len(X_train),
        tuning_rows=len(X_tune),
    )

    with mlflow.start_run(run_name="champion_selection"):
        mlflow.set_tag("problem_type", problem_type)
        mlflow.set_tag("champion_model", champion["model_name"])
        mlflow.log_metrics(
            {
                "champion_val_accuracy_pct": champion["validation_accuracy_pct"],
                "champion_train_accuracy_pct": champion["training_accuracy_pct"],
                "champion_gap_pct": champion["accuracy_gap_pct"],
                "champion_size_mb": champion["model_size_mb"],
                "champion_train_time_s": champion["training_time_seconds"],
                "champion_pred_time_ms": champion["prediction_time_ms"],
                "champion_selection_cost": champion["selection_cost"],
            }
        )
        mlflow.log_artifact(str(SELECTION_REPORT_PATH))
        mlflow.log_artifact(str(COMPARISON_CSV_PATH))
        mlflow.log_artifact(str(COMPARISON_MD_PATH))

    print(
        f"\n[Champion Selected] {champion['model_name']} -> {CHAMPION_MODEL_PATH} "
        f"(Val Acc: {champion['validation_accuracy_pct']:.2f}%, Cost: {champion['selection_cost']:.4f})"
    )


if __name__ == "__main__":
    main()
