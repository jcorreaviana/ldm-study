"""
2. Avalia o baseline XGBoost no conjunto de teste.

Gera: precision/recall/F1 em múltiplos thresholds, PR-AUC, ROC-AUC,
matriz de confusão, curva precision-recall, curva ROC e feature importance.

Métricas de acurácia bruta são ignoradas de propósito: com ~0,7% de fraude,
um modelo que nunca prevê fraude já acerta 99,3% — por isso o foco é
precision/recall/PR-AUC.
"""

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from utils import OUT_DIR


def main():
    with open(OUT_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_parquet(OUT_DIR / "X_test.parquet")
    y_test = pd.read_parquet(OUT_DIR / "y_test.parquet")["isFraud"]
    meta_test = pd.read_parquet(OUT_DIR / "meta_test.parquet")

    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("=" * 60)
    print("MÉTRICAS GLOBAIS (conjunto de teste)")
    print("=" * 60)
    print(f"n teste: {len(y_test):,} | fraudes: {int(y_test.sum())}".replace(",", "."))
    print(f"PR-AUC (average precision): {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # métricas em alguns thresholds de decisão
    print()
    print("=" * 60)
    print("PRECISION / RECALL / F1 EM DIFERENTES THRESHOLDS")
    print("=" * 60)
    resultados_threshold = []
    for th in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        y_pred = (y_proba >= th).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        n_alertas = int(y_pred.sum())
        print(f"threshold={th:.1f} | precision={p:.3f} | recall={r:.3f} | f1={f1:.3f} | alertas={n_alertas}")
        resultados_threshold.append(
            {"threshold": th, "precision": p, "recall": r, "f1": f1, "n_alertas": n_alertas}
        )

    # threshold 0.5 como referência para matriz de confusão e classification_report
    y_pred_05 = (y_proba >= 0.5).astype(int)
    print()
    print("=" * 60)
    print("CLASSIFICATION REPORT (threshold=0.5)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_05, target_names=["Legítima", "Fraude"], digits=3))

    cm = confusion_matrix(y_test, y_pred_05)
    print("Matriz de confusão (threshold=0.5):")
    print(cm)

    # --- gráficos ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # matriz de confusão
    ax = axes[0, 0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legítima", "Fraude"])
    ax.set_yticklabels(["Legítima", "Fraude"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão (threshold=0.5)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    # curva PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    baseline_rate = y_test.mean()
    ax = axes[0, 1]
    ax.plot(recall, precision, color="#C44E52", label=f"XGBoost (PR-AUC={pr_auc:.3f})")
    ax.axhline(baseline_rate, color="gray", linestyle="--", label=f"Aleatório (taxa base={baseline_rate:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend()

    # curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax = axes[1, 0]
    ax.plot(fpr, tpr, color="#4C72B0", label=f"XGBoost (ROC-AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Aleatório")
    ax.set_xlabel("Falso Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend()

    # feature importance
    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=True)
    ax = axes[1, 1]
    ax.barh(importances.index, importances.values, color="#55A868")
    ax.set_title("Importância das features (XGBoost)")
    ax.set_xlabel("Importância (gain normalizado)")

    plt.tight_layout()
    out_path = OUT_DIR / "02_avaliacao_baseline.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {out_path}")

    # salva métricas em json para consulta posterior
    metrics_out = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "n_teste": int(len(y_test)),
        "n_fraudes_teste": int(y_test.sum()),
        "thresholds": resultados_threshold,
        "confusion_matrix_threshold_0.5": cm.tolist(),
        "feature_importance": importances.sort_values(ascending=False).to_dict(),
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Métricas salvas em: {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
