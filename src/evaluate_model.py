"""
evaluate_model.py - Model Evaluation & Visualization
=====================================================
Evaluates trained models with comprehensive metrics and generates
publication-quality visualizations: confusion matrices, ROC curves,
feature importance charts, and comparison reports.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/Colab compatibility
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    CONFIG, setup_logger, ensure_directories, save_json,
    MODELS_DIR, REPORTS_DIR,
)

logger = setup_logger(__name__)

# ─────────────────────── Matplotlib Style ────────────────────────────

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.family": "sans-serif",
})

COLORS = {
    "Logistic_Regression": "#3498db",
    "Random_Forest": "#2ecc71",
    "XGBoost": "#e74c3c",
    "SVM": "#9b59b6",
    "MLP_Classifier": "#f39c12",
}


# ─────────────────────── Metrics Computation ─────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    dict : {metric_name: value}
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)

    return metrics


# ─────────────────────── Evaluate All Models ─────────────────────────

def evaluate_all_models(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all trained models on the test set.

    Parameters
    ----------
    trained_models : dict
        {name: {"model": model, ...}}
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.

    Returns
    -------
    pd.DataFrame : Comparison table of all model metrics.
    """
    ensure_directories()
    results = []

    logger.info("=" * 70)
    logger.info("  MODEL EVALUATION ON TEST SET")
    logger.info("=" * 70)

    for name, info in trained_models.items():
        model = info["model"]
        y_pred = model.predict(X_test.values)

        # Get probabilities if available
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test.values)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test.values)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["model"] = name
        metrics["train_time_s"] = info.get("train_time", 0)
        results.append(metrics)

        logger.info(
            f"  {name:<22s} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"AUC: {metrics.get('roc_auc', 'N/A')}"
        )

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    cols_order = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "train_time_s"]
    results_df = results_df[[c for c in cols_order if c in results_df.columns]]
    results_df = results_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)

    # Save results
    results_path = str(REPORTS_DIR / "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # Save as JSON too
    save_json(results_df.to_dict(orient="records"), str(REPORTS_DIR / "model_comparison.json"))

    return results_df


# ─────────────────────── Confusion Matrix ────────────────────────────

def plot_confusion_matrices(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Generate and save confusion matrices for all models.
    """
    ensure_directories()
    fig_dir = REPORTS_DIR / "figures"
    n_models = len(trained_models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, info) in enumerate(trained_models.items()):
        model = info["model"]
        y_pred = model.predict(X_test.values)
        cm = confusion_matrix(y_test, y_pred)

        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(name.replace("_", " "), fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_xticklabels(["No Disease", "Disease"])
        ax.set_yticklabels(["No Disease", "Disease"])

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Confusion Matrices — Heart Disease Prediction", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = str(fig_dir / "confusion_matrices.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrices saved to: {save_path}")


# ─────────────────────── ROC Curves ──────────────────────────────────

def plot_roc_curves(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Generate and save a combined ROC curve plot for all models.
    """
    ensure_directories()
    fig_dir = REPORTS_DIR / "figures"

    fig, ax = plt.subplots(figsize=(10, 8))

    for name, info in trained_models.items():
        model = info["model"]
        color = COLORS.get(name, "#333333")

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test.values)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test.values)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        ax.plot(
            fpr, tpr,
            label=f"{name.replace('_', ' ')} (AUC = {auc:.4f})",
            color=color,
            linewidth=2.5,
            alpha=0.9,
        )

    # Reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Random (AUC = 0.5)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = str(fig_dir / "roc_curves.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"ROC curves saved to: {save_path}")


# ─────────────────────── Feature Importance ──────────────────────────

def plot_feature_importance(
    trained_models: dict,
    feature_names: list,
    top_n: int = 20,
) -> None:
    """
    Plot feature importance from tree-based models (Random Forest, XGBoost).
    """
    ensure_directories()
    fig_dir = REPORTS_DIR / "figures"

    tree_models = {
        "Random_Forest": trained_models.get("Random_Forest"),
        "XGBoost": trained_models.get("XGBoost"),
    }

    for name, info in tree_models.items():
        if info is None:
            continue

        model = info["model"]

        if not hasattr(model, "feature_importances_"):
            continue

        importances = model.feature_importances_

        # Handle case where feature_names length doesn't match
        if len(importances) != len(feature_names):
            feat_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feat_names = feature_names

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feat_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(12, 8))
        color = COLORS.get(name, "#3498db")

        bars = ax.barh(
            range(len(top_features)),
            top_importances,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(
            f"Top {top_n} Feature Importances — {name.replace('_', ' ')}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, top_importances):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        save_path = str(fig_dir / f"feature_importance_{name.lower()}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Feature importance plot saved: {save_path}")

    # Save feature importances as CSV
    if "Random_Forest" in trained_models:
        rf_model = trained_models["Random_Forest"]["model"]
        if hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            if len(importances) == len(feature_names):
                fi_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": importances,
                }).sort_values("importance", ascending=False)
                fi_df.to_csv(str(REPORTS_DIR / "feature_importances.csv"), index=False)


# ─────────────────────── Metrics Comparison Bar Chart ────────────────

def plot_metrics_comparison(results_df: pd.DataFrame) -> None:
    """
    Create a grouped bar chart comparing all metrics across models.
    """
    ensure_directories()
    fig_dir = REPORTS_DIR / "figures"

    metrics_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    available_metrics = [c for c in metrics_cols if c in results_df.columns]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(results_df))
    width = 0.15
    multiplier = 0

    for metric in available_metrics:
        offset = width * multiplier
        bars = ax.bar(
            x + offset,
            results_df[metric],
            width,
            label=metric.replace("_", " ").title(),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )
        multiplier += 1

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
    ax.set_xticklabels(
        [m.replace("_", " ") for m in results_df["model"]],
        fontsize=11,
        rotation=15,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = str(fig_dir / "metrics_comparison.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Metrics comparison chart saved: {save_path}")


# ─────────────────────── Classification Reports ─────────────────────

def generate_classification_reports(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Generate and save detailed classification reports for all models.
    """
    ensure_directories()

    reports = {}
    for name, info in trained_models.items():
        model = info["model"]
        y_pred = model.predict(X_test.values)

        report = classification_report(
            y_test, y_pred,
            target_names=["No Disease", "Heart Disease"],
            output_dict=True,
        )
        reports[name] = report

    save_json(reports, str(REPORTS_DIR / "classification_reports.json"))
    logger.info("Classification reports saved to reports/classification_reports.json")


# ─────────────────────── Full Evaluation Pipeline ────────────────────

def run_evaluation(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list = None,
) -> pd.DataFrame:
    """
    Execute the complete evaluation pipeline:

    1. Compute metrics for all models
    2. Plot confusion matrices
    3. Plot ROC curves
    4. Plot feature importance (tree-based models)
    5. Plot metrics comparison chart
    6. Generate classification reports
    7. Print final summary

    Returns
    -------
    pd.DataFrame : Model comparison results.
    """
    logger.info("=" * 70)
    logger.info("  EVALUATION PIPELINE STARTED")
    logger.info("=" * 70)

    # 1. Metrics
    results_df = evaluate_all_models(trained_models, X_test, y_test)

    # 2. Confusion Matrices
    plot_confusion_matrices(trained_models, X_test, y_test)

    # 3. ROC Curves
    plot_roc_curves(trained_models, X_test, y_test)

    # 4. Feature Importance
    if feature_names:
        plot_feature_importance(trained_models, feature_names)

    # 5. Metrics Comparison
    plot_metrics_comparison(results_df)

    # 6. Classification Reports
    generate_classification_reports(trained_models, X_test, y_test)

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("  EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\n{results_df.to_string(index=False)}")
    logger.info("\n📁 Reports saved to: reports/")
    logger.info("📊 Figures saved to: reports/figures/")
    logger.info("=" * 70)

    return results_df


# ─────────────────────── CLI Entry Point ─────────────────────────────

if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline
    from src.feature_engineering import run_feature_engineering
    from src.train_model import train_all_models, select_best_model

    # Full pipeline
    X_train, X_test, y_train, y_test, features = run_preprocessing_pipeline()
    X_train_final, X_test_final = run_feature_engineering(X_train, X_test)

    trained_models = train_all_models(X_train_final.values, y_train.values)
    best_name, best_model = select_best_model(trained_models, X_train_final.values, y_train.values)

    feature_names = X_train_final.columns.tolist()
    results_df = run_evaluation(trained_models, X_test_final, y_test, feature_names)

    print(f"\n✅ Evaluation complete! Check reports/ directory.")
