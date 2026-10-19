from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_prep import load_and_prepare_dataset  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "best_model.joblib"


def make_pipeline(model_name: str) -> Pipeline:
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            C=2.0,
            solver="liblinear",
            random_state=42,
        )
    elif model_name == "multinomial_nb":
        classifier = MultinomialNB(alpha=0.5)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=150_000,
            sublinear_tf=True,
        )),
        ("classifier", classifier),
    ])


def calculate_metrics(model: Pipeline, X, y, split_name: str) -> dict:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    result = {
        "split": split_name,
        "accuracy": float(accuracy_score(y, predictions)),
        "fraud_precision": float(precision_score(y, predictions, zero_division=0)),
        "fraud_recall": float(recall_score(y, predictions, zero_division=0)),
        "fraud_f1": float(f1_score(y, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if y.nunique() == 2 else None,
        "classification_report": classification_report(
            y, predictions, labels=[0, 1],
            target_names=["legitimate", "fraudulent"],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix_labels": ["legitimate", "fraudulent"],
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }
    return result


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, data_summary = load_and_prepare_dataset()

    if y.value_counts().min() < 5:
        raise ValueError("Too few examples in one class after deduplication to create reliable splits.")

    # The source dataset has no trustworthy timestamp field. Therefore we use a
    # stratified holdout rather than pretending this is a temporal split.
    # Exact duplicate postings were removed before splitting. The TF-IDF vocabulary
    # and IDF values are learned only inside the training pipeline.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1875, random_state=42, stratify=y_trainval
    )
    # Overall split is approximately 65% train, 15% validation, 20% test.
    # Test data is not used to choose the model.
    split_summary = {
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "strategy": "stratified random split after exact duplicate removal; dataset does not provide a usable posting timestamp",
        "random_state": 42,
    }

    candidates = {}
    validation_results = {}
    for model_name in ["logistic_regression", "multinomial_nb"]:
        print(f"\nTraining {model_name}...")
        model = make_pipeline(model_name)
        model.fit(X_train["job_text"], y_train)
        metrics = calculate_metrics(model, X_val["job_text"], y_val, "validation")
        candidates[model_name] = model
        validation_results[model_name] = metrics
        print(
            f"Validation fraud F1={metrics['fraud_f1']:.3f} | "
            f"precision={metrics['fraud_precision']:.3f} | "
            f"recall={metrics['fraud_recall']:.3f} | "
            f"average_precision={metrics['average_precision']:.3f}"
        )

    # Select on validation fraud-class F1; use average precision as a tie-breaker.
    selected_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["fraud_f1"],
            validation_results[name]["average_precision"],
        ),
    )
    print(f"\nSelected model based on validation metrics: {selected_name}")

    # Refit the selected pipeline on train + validation, then evaluate test exactly once.
    X_trainval_text = X_trainval["job_text"]
    selected_model = make_pipeline(selected_name)
    selected_model.fit(X_trainval_text, y_trainval)
    test_metrics = calculate_metrics(selected_model, X_test["job_text"], y_test, "test")

    joblib.dump(selected_model, MODEL_PATH)

    # Save a confusion matrix plot for the final test evaluation.
    cm = test_metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm)
    ax.set_title("Career Opportunity Checker — Test Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], labels=["Legitimate", "Fraudulent"])
    ax.set_yticks([0, 1], labels=["Legitimate", "Fraudulent"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    report = {
        "project": "Career Opportunity Checker",
        "dataset_summary": data_summary,
        "split_summary": split_summary,
        "validation_model_comparison": {
            name: {
                key: metrics[key]
                for key in ["accuracy", "fraud_precision", "fraud_recall", "fraud_f1", "average_precision", "roc_auc"]
            }
            for name, metrics in validation_results.items()
        },
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "caution": (
            "Historical benchmark only. The dataset has no usable timestamp in this copy, "
            "so results use a stratified holdout, not a future-time evaluation. Do not treat "
            "predictions as proof of fraud or assume current real-world performance."
        ),
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nFINAL TEST RESULTS (used once after model selection)")
    for key in ["accuracy", "fraud_precision", "fraud_recall", "fraud_f1", "average_precision", "roc_auc"]:
        value = test_metrics[key]
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: unavailable")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_DIR / 'metrics.json'}")
    print(f"Saved plot: {REPORT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
