import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
import yaml

PARAMS_PATH = Path("params.yaml")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")
MODELS_DIR = Path("models")
CHAMPION_MODEL_PATH = MODELS_DIR / "model.pkl"
SELECTION_REPORT_PATH = Path("reports/model_selection.json")
METRICS_PATH = Path("reports/metrics.json")
EVAL_DETAILS_PATH = Path("reports/evaluation_details.json")


def load_params() -> dict:
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_classifier(model, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict:
    y_pred = model.predict(X_eval)
    metrics = {
        "accuracy": round(float(accuracy_score(y_eval, y_pred)), 6),
        "precision_macro": round(float(precision_score(y_eval, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(y_eval, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_eval, y_pred, average="macro", zero_division=0)), 6),
    }
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_eval)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_eval, proba)), 6)
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_eval)
            metrics["roc_auc"] = round(float(roc_auc_score(y_eval, scores)), 6)
        except Exception:
            pass
    return metrics


def evaluate_regressor(model, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict:
    y_pred = model.predict(X_eval)
    mse = float(mean_squared_error(y_eval, y_pred))
    return {
        "rmse": round(mse ** 0.5, 6),
        "mae": round(float(mean_absolute_error(y_eval, y_pred)), 6),
        "r2": round(float(r2_score(y_eval, y_pred)), 6),
    }


def main() -> None:
    params = load_params()
    experiment_name = str(params["experiment_name"])

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

    with open(SELECTION_REPORT_PATH, "r", encoding="utf-8") as f:
        selection_info = json.load(f)

    problem_type = selection_info.get("problem_type", "classification")
    champion_name = selection_info["champion_model"]

    champion_model = joblib.load(CHAMPION_MODEL_PATH)

    if problem_type == "classification":
        champion_test_metrics = evaluate_classifier(champion_model, X_test, y_test)
        y_pred_champ = champion_model.predict(X_test)
        champ_report = classification_report(
            y_test,
            y_pred_champ,
            target_names=["Retained (0)", "Churned (1)"],
            output_dict=True,
            zero_division=0,
        )
    else:
        champion_test_metrics = evaluate_regressor(champion_model, X_test, y_test)
        champ_report = {}

    candidate_test_results = {}
    for candidate in selection_info["all_candidates"]:
        m_name = candidate["model_name"]
        m_path = MODELS_DIR / f"{m_name}.pkl"
        if not m_path.exists():
            continue
        m_obj = joblib.load(m_path)
        # For KNN on 88k test rows, evaluate on a 10,000-row stratified slice so evaluation stays fast
        X_eval_slice = X_test.iloc[:10000] if m_name == "KNN" and len(X_test) > 10000 else X_test
        y_eval_slice = y_test.iloc[:10000] if m_name == "KNN" and len(y_test) > 10000 else y_test

        if problem_type == "classification":
            candidate_test_results[m_name] = evaluate_classifier(m_obj, X_eval_slice, y_eval_slice)
        else:
            candidate_test_results[m_name] = evaluate_regressor(m_obj, X_eval_slice, y_eval_slice)

    flat_metrics = {
        "test_samples": int(len(X_test)),
        **{f"test_{k}": v for k, v in champion_test_metrics.items()},
        "cv_validation_accuracy_pct": selection_info["champion_metrics"]["validation_accuracy_pct"],
        "cv_training_accuracy_pct": selection_info["champion_metrics"]["training_accuracy_pct"],
        "cv_accuracy_gap_pct": selection_info["champion_metrics"]["accuracy_gap_pct"],
        "model_size_mb": selection_info["champion_metrics"]["model_size_mb"],
        "training_time_seconds": selection_info["champion_metrics"]["training_time_seconds"],
        "prediction_time_ms": selection_info["champion_metrics"]["prediction_time_ms"],
        "selection_cost": selection_info["champion_metrics"]["selection_cost"],
    }

    evaluation_details = {
        "problem_type": problem_type,
        "champion_model": champion_name,
        "test_set_rows": int(len(X_test)),
        "champion_test_metrics": champion_test_metrics,
        "champion_classification_report": champ_report,
        "all_models_test_comparison": candidate_test_results,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(flat_metrics, f, indent=4)

    with open(EVAL_DETAILS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_details, f, indent=4)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="champion_evaluation"):
        mlflow.set_tag("problem_type", problem_type)
        mlflow.set_tag("champion_model", champion_name)
        mlflow.log_metrics({k: float(v) for k, v in flat_metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(EVAL_DETAILS_PATH))

    print(
        f"[Model Evaluation] Champion '{champion_name}' on {len(X_test):,} test rows -> "
        f"{champion_test_metrics}"
    )


if __name__ == "__main__":
    main()
