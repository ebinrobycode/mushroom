"""
Model Training, Cross-Validation & Evaluation Pipeline for Mushroom Classification.
Focus Algorithm: Naive Bayes Classifier (BernoulliNB, MultinomialNB, GaussianNB, and baseline DecisionTree).
Saves model.pkl, metadata, and publication-ready evaluation charts.
"""

import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.autolayout'] = True

CLASS_NAMES = ["Edible", "Poisonous"]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset.csv")
    model_output_path = os.path.join(base_dir, "model.pkl")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("==================================================")
    print("MUSHROOM CLASSIFICATION: NAIVE BAYES BENCHMARK")
    print("==================================================")

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset shape: {df.shape}")

    feature_cols = [
        "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
        "gill_attachment", "gill_spacing", "gill_size", "gill_color",
        "stalk_surface_above_ring", "ring_type", "spore_print_color",
        "population", "habitat"
    ]
    target_col = "edibility"

    X = df[feature_cols]
    y = df[target_col]

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training records: {len(X_train)} | Test records: {len(X_test)}")
    print("\nTraining Class Distribution:")
    print(y_train.value_counts())

    preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    models = {
        "Bernoulli Naive Bayes": BernoulliNB(alpha=1.0),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree (Baseline)": DecisionTreeClassifier(max_depth=6, random_state=42)
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_test_bin = (y_test == "Poisonous").astype(int)

    fig_cm, axes_cm = plt.subplots(1, 4, figsize=(22, 5))
    cm_idx = 0

    for name, clf in models.items():
        print(f"\nEvaluating: {name}...")
        pipe = Pipeline(steps=[
            ("encoder", preprocessor),
            ("classifier", clf)
        ])

        # 5-fold cross validation on training set
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        print(f"  5-Fold CV Accuracy:  {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
        print(f"  5-Fold CV Macro F1:  {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")

        # Fit on full training data
        pipe.fit(X_train, y_train)

        # Test set prediction
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(y_test_bin, y_proba) if y_proba is not None else None

        print(f"  Test Accuracy:         {acc:.4f}")
        print(f"  Test Precision (Macro):{prec_macro:.4f}")
        print(f"  Test Recall (Macro):   {rec_macro:.4f}")
        print(f"  Test F1-Score (Macro): {f1_macro:.4f}")
        print(f"  Test F1-Score (Weight):{f1_weighted:.4f}")
        if roc_auc:
            print(f"  Test ROC-AUC:          {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=axes_cm[cm_idx],
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        axes_cm[cm_idx].set_title(f"{name}\n(Acc: {acc*100:.2f}%, F1: {f1_macro:.3f})", fontsize=10, fontweight="bold")
        axes_cm[cm_idx].set_ylabel("True Category" if cm_idx == 0 else "")
        axes_cm[cm_idx].set_xlabel("Predicted Category")
        cm_idx += 1

        results[name] = {
            "pipeline": pipe,
            "cv_accuracy_mean": float(cv_acc.mean()),
            "cv_accuracy_std": float(cv_acc.std()),
            "cv_f1_macro_mean": float(cv_f1.mean()),
            "test_accuracy": float(acc),
            "test_precision_macro": float(prec_macro),
            "test_recall_macro": float(rec_macro),
            "test_f1_macro": float(f1_macro),
            "test_f1_weighted": float(f1_weighted),
            "test_roc_auc": float(roc_auc) if roc_auc else None,
            "confusion_matrix": cm.tolist(),
            "y_pred": y_pred,
            "y_proba": y_proba
        }

    fig_cm.suptitle("Confusion Matrix Comparison - Mushroom Edibility Classifiers", fontsize=14, fontweight="bold", y=1.03)
    fig_cm.savefig(os.path.join(reports_dir, "confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_cm)
    print("Saved confusion_matrices.png")

    # 1. Class Distribution Chart
    plt.figure(figsize=(7, 4.5))
    class_counts = df[target_col].value_counts()[CLASS_NAMES]
    colors = ["#16A34A", "#DC2626"]
    bars = plt.bar(class_counts.index, class_counts.values, color=colors, edgecolor="#0f172a", linewidth=1.2, width=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 35,
                 f"{h} ({h/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0f172a")
    plt.title("Mushroom Dataset Class Balance (N=4,000)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Number of Specimens", fontsize=11)
    plt.ylim(0, max(class_counts.values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(reports_dir, "class_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved class_distribution.png")

    # 2. Model Performance Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    m_names = list(results.keys())
    accs = [results[m]["test_accuracy"] * 100 for m in m_names]
    f1s = [results[m]["test_f1_macro"] * 100 for m in m_names]
    cv_accs = [results[m]["cv_accuracy_mean"] * 100 for m in m_names]

    x = np.arange(len(m_names))
    width = 0.25

    plt.bar(x - width, accs, width, label="Test Accuracy (%)", color="#1E3A8A", edgecolor="#0f172a")
    plt.bar(x, f1s, width, label="Test Macro F1 (%)", color="#16A34A", edgecolor="#14532D")
    plt.bar(x + width, cv_accs, width, label="5-Fold CV Accuracy (%)", color="#38BDF8", edgecolor="#0284C7")

    plt.ylabel("Performance (%)", fontsize=11, fontweight="bold")
    plt.title("Model Performance Comparison (Accuracy, F1-Score, 5-Fold CV)", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(x, m_names, fontsize=9.5, fontweight="bold")
    plt.ylim(80, 105)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(reports_dir, "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved model_comparison.png")

    # 3. ROC Curves for Models
    plt.figure(figsize=(8, 6))
    curve_colors = ["#16A34A", "#2563EB", "#D97706", "#9333EA"]

    for idx, (m_name, res) in enumerate(results.items()):
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test_bin, res["y_proba"])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=curve_colors[idx], lw=2.5,
                     label=f"{m_name} (AUC = {roc_auc_val:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("ROC Curves Comparison - Mushroom Edibility", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(reports_dir, "roc_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved roc_curves.png")

    # 4. Feature Importance / Conditional Log Likelihood for BernoulliNB
    best_pipe = results["Bernoulli Naive Bayes"]["pipeline"]
    bnb_clf = best_pipe.named_steps["classifier"]
    encoder = best_pipe.named_steps["encoder"]

    encoded_feature_names = encoder.get_feature_names_out(feature_cols)
    # Difference in log probability between Poisonous and Edible indicates strongest discriminators
    feature_log_diff = np.abs(bnb_clf.feature_log_prob_[1] - bnb_clf.feature_log_prob_[0])

    feat_df = pd.DataFrame({"Feature": encoded_feature_names, "Discriminative_Power": feature_log_diff})
    feat_df = feat_df.sort_values(by="Discriminative_Power", ascending=True).tail(12)

    plt.figure(figsize=(9, 6))
    plt.barh(feat_df["Feature"], feat_df["Discriminative_Power"], color="#16A34A", edgecolor="#14532D")
    plt.xlabel("Log-Probability Absolute Divergence (|log P(x|Poisonous) - log P(x|Edible)|)", fontsize=10, fontweight="bold")
    plt.title("Naive Bayes - Top Morphological Discriminators", fontsize=13, fontweight="bold", pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(reports_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved feature_importance.png")

    # Primary algorithm to save: Bernoulli Naive Bayes
    winner_name = "Bernoulli Naive Bayes"
    winner_pipeline = results[winner_name]["pipeline"]
    joblib.dump(winner_pipeline, model_output_path)
    print(f"\nSuccessfully serialized {winner_name} to: {model_output_path}")

    # Save metadata JSON for reporting & docs
    metadata = {
        "primary_algorithm": winner_name,
        "algorithm_family": "Naive Bayes Classifier",
        "classes": CLASS_NAMES,
        "input_features": feature_cols,
        "model_comparison": {
            name: {
                "cv_accuracy_mean": round(res["cv_accuracy_mean"], 4),
                "cv_accuracy_std": round(res["cv_accuracy_std"], 4),
                "cv_f1_macro_mean": round(res["cv_f1_macro_mean"], 4),
                "test_accuracy": round(res["test_accuracy"], 4),
                "test_precision_macro": round(res["test_precision_macro"], 4),
                "test_recall_macro": round(res["test_recall_macro"], 4),
                "test_f1_macro": round(res["test_f1_macro"], 4),
                "test_f1_weighted": round(res["test_f1_weighted"], 4),
                "test_roc_auc": round(res["test_roc_auc"], 4) if res["test_roc_auc"] else None,
                "confusion_matrix": res["confusion_matrix"]
            }
            for name, res in results.items()
        }
    }
    with open(os.path.join(reports_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n==================================================")
    print(f"TRAINING COMPLETE! Winner: {winner_name}")
    print(f"Test Accuracy:  {results[winner_name]['test_accuracy'] * 100:.2f}%")
    print(f"Test Macro F1:  {results[winner_name]['test_f1_macro'] * 100:.2f}%")
    print(f"Test ROC-AUC:   {results[winner_name]['test_roc_auc']:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
