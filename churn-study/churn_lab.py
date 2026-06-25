"""
╔══════════════════════════════════════════════════════════════════╗
║              CHURN LAB — Telco Customer Churn                   ║
║   Modelos clássicos vs TabTransformer (proxy LDM)               ║
║   Métricas: Matriz de confusão, AUROC, AUPRC, Precision, Recall ║
╚══════════════════════════════════════════════════════════════════╝

Estrutura:
  BLOCO 1 — Carregar e explorar os dados
  BLOCO 2 — Pré-processamento
  BLOCO 3 — Regressão Logística
  BLOCO 4 — XGBoost
  BLOCO 5 — Random Forest
  BLOCO 6 — TabTransformer (proxy LDM)
  BLOCO 7 — Visualizações comparativas
"""

# ─────────────────────────────────────────────
# DEPENDÊNCIAS
# pip install pandas numpy scikit-learn xgboost matplotlib seaborn
# pip install torch pytorch-tabnet
# ─────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)

import xgboost as xgb

# ─── Utilitários de progresso ───
def bloco(n, titulo):
    print()
    print("═" * 60)
    print(f"  BLOCO {n} — {titulo}")
    print("═" * 60)

def ok(msg, start=None):
    if start:
        elapsed = time.time() - start
        print(f"  ✅ {msg} ({elapsed:.1f}s)")
    else:
        print(f"  ✅ {msg}")

def info(msg):
    print(f"  ℹ️  {msg}")

def metrica(auroc, auprc, prec, rec, f1):
    print(f"  {'AUROC':<12} {auroc:.4f}")
    print(f"  {'AUPRC':<12} {auprc:.4f}")
    print(f"  {'Precision':<12} {prec:.4f}")
    print(f"  {'Recall':<12} {rec:.4f}")
    print(f"  {'F1':<12} {f1:.4f}")

# Estilo visual
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "logistic":      "#4f8fff",
    "xgboost":       "#3ecf8e",
    "random_forest": "#ef9f27",
    "tabtransformer":"#e24b4a",
    "baseline":      "#888888",
}

print("✅ Dependências carregadas.")

# ══════════════════════════════════════════════════════════════════
# BLOCO 1 — CARREGAR E EXPLORAR OS DADOS
# ══════════════════════════════════════════════════════════════════
bloco(1, "Carregando os dados")

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

print("  ⏳ Baixando dataset...")
t = timer_start = time.time()
try:
    df = pd.read_csv(URL)
    ok(f"Dataset carregado: {df.shape[0]} clientes, {df.shape[1]} colunas", t)
except Exception as e:
    print(f"  ❌ Erro ao baixar: {e}")
    print("     Baixe manualmente: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    raise

print()
print("  ── Primeiras 3 linhas ──")
print(df.head(3).to_string())
print()
print("  ── Distribuição de Churn ──")
print(df["Churn"].value_counts().to_string())
print(f"\n  Taxa de churn: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")

# ══════════════════════════════════════════════════════════════════
# BLOCO 2 — PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════
bloco(2, "Pré-processamento")

t = time.time()
print("  ⏳ Processando...")

df2 = df.copy()
df2 = df2.drop(columns=["customerID"])
df2["TotalCharges"] = pd.to_numeric(df2["TotalCharges"], errors="coerce")
df2["TotalCharges"] = df2["TotalCharges"].fillna(df2["TotalCharges"].median())
df2["Churn"] = (df2["Churn"] == "Yes").astype(int)

X = df2.drop(columns=["Churn"])
y = df2["Churn"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

ok("Pré-processamento concluído", t)
info(f"Features numéricas ({len(num_cols)}): {num_cols}")
info(f"Features categóricas ({len(cat_cols)}): {cat_cols}")
info(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
info(f"Positivos no teste (churn=1): {y_test.sum()} ({y_test.mean():.1%})")

# ══════════════════════════════════════════════════════════════════
# BLOCO 3 — REGRESSÃO LOGÍSTICA
# ══════════════════════════════════════════════════════════════════
bloco(3, "Regressão Logística")

print("  ⏳ Treinando...")
t = time.time()

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_pred_lr  = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

auroc_lr = roc_auc_score(y_test, y_proba_lr)
auprc_lr = average_precision_score(y_test, y_proba_lr)
prec_lr  = precision_score(y_test, y_pred_lr)
rec_lr   = recall_score(y_test, y_pred_lr)
f1_lr    = f1_score(y_test, y_pred_lr)

ok("Regressão Logística treinada", t)
metrica(auroc_lr, auprc_lr, prec_lr, rec_lr, f1_lr)

# ══════════════════════════════════════════════════════════════════
# BLOCO 4 — XGBOOST
# ══════════════════════════════════════════════════════════════════
bloco(4, "XGBoost")

print("  ⏳ Treinando...")
t = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)

y_pred_xgb  = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

auroc_xgb = roc_auc_score(y_test, y_proba_xgb)
auprc_xgb = average_precision_score(y_test, y_proba_xgb)
prec_xgb  = precision_score(y_test, y_pred_xgb)
rec_xgb   = recall_score(y_test, y_pred_xgb)
f1_xgb    = f1_score(y_test, y_pred_xgb)

ok("XGBoost treinado", t)
metrica(auroc_xgb, auprc_xgb, prec_xgb, rec_xgb, f1_xgb)

# ══════════════════════════════════════════════════════════════════
# BLOCO 5 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════
bloco(5, "Random Forest")

print("  ⏳ Treinando...")
t = time.time()

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf  = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

auroc_rf = roc_auc_score(y_test, y_proba_rf)
auprc_rf = average_precision_score(y_test, y_proba_rf)
prec_rf  = precision_score(y_test, y_pred_rf)
rec_rf   = recall_score(y_test, y_pred_rf)
f1_rf    = f1_score(y_test, y_pred_rf)

ok("Random Forest treinado", t)
metrica(auroc_rf, auprc_rf, prec_rf, rec_rf, f1_rf)

# ══════════════════════════════════════════════════════════════════
# BLOCO 6 — TABTRANSFORMER (proxy LDM)
# ══════════════════════════════════════════════════════════════════
bloco(6, "TabTransformer (proxy LDM)")

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch

    info(f"PyTorch {torch.__version__} detectado")
    info("Parâmetros: 50 épocas máx, patience=10, batch=512 (otimizado para CPU)")
    info("Iniciando treinamento...")
    print()

    X_train_np = X_train.values.astype(np.float32)
    X_test_np  = X_test.values.astype(np.float32)
    y_train_np = y_train.values
    y_test_np  = y_test.values

    t = time.time()

    tab = TabNetClassifier(
        n_d=8, n_a=8,
        n_steps=2,
        gamma=1.3,
        n_independent=1,
        n_shared=1,
        momentum=0.02,
        mask_type="sparsemax",
        verbose=10,
        seed=42
    )

    tab.fit(
        X_train_np, y_train_np,
        eval_set=[(X_test_np, y_test_np)],
        eval_metric=["auc"],
        max_epochs=50,
        patience=10,
        batch_size=512,
        virtual_batch_size=256,
    )

    y_proba_tab = tab.predict_proba(X_test_np)[:, 1]
    y_pred_tab  = (y_proba_tab >= 0.5).astype(int)

    auroc_tab = roc_auc_score(y_test, y_proba_tab)
    auprc_tab = average_precision_score(y_test, y_proba_tab)
    prec_tab  = precision_score(y_test, y_pred_tab)
    rec_tab   = recall_score(y_test, y_pred_tab)
    f1_tab    = f1_score(y_test, y_pred_tab)
    tab_available = True

    print()
    ok("TabNet treinado", t)
    metrica(auroc_tab, auprc_tab, prec_tab, rec_tab, f1_tab)

except ImportError:
    print("  ⚠️  pytorch-tabnet não instalado.")
    print("     Execute: pip install pytorch-tabnet torch")
    tab_available = False
    y_proba_tab = y_proba_xgb
    auroc_tab = auroc_xgb
    auprc_tab = auprc_xgb
    prec_tab  = prec_xgb
    rec_tab   = rec_xgb
    f1_tab    = f1_xgb

# ══════════════════════════════════════════════════════════════════
# BLOCO 7 — VISUALIZAÇÕES COMPARATIVAS
# ══════════════════════════════════════════════════════════════════
bloco(7, "Gerando visualizações")

models = {
    "Regressão Logística": {
        "color": COLORS["logistic"],
        "y_pred": y_pred_lr,
        "y_proba": y_proba_lr,
        "auroc": auroc_lr, "auprc": auprc_lr,
        "prec": prec_lr, "rec": rec_lr, "f1": f1_lr,
    },
    "XGBoost": {
        "color": COLORS["xgboost"],
        "y_pred": y_pred_xgb,
        "y_proba": y_proba_xgb,
        "auroc": auroc_xgb, "auprc": auprc_xgb,
        "prec": prec_xgb, "rec": rec_xgb, "f1": f1_xgb,
    },
    "Random Forest": {
        "color": COLORS["random_forest"],
        "y_pred": y_pred_rf,
        "y_proba": y_proba_rf,
        "auroc": auroc_rf, "auprc": auprc_rf,
        "prec": prec_rf, "rec": rec_rf, "f1": f1_rf,
    },
    "TabTransformer (LDM proxy)": {
        "color": COLORS["tabtransformer"],
        "y_pred": (y_proba_tab >= 0.5).astype(int),
        "y_proba": y_proba_tab,
        "auroc": auroc_tab, "auprc": auprc_tab,
        "prec": prec_tab, "rec": rec_tab, "f1": f1_tab,
    },
}

# ── FIGURA 1: Matrizes de confusão ──
print("  ⏳ Gerando fig1 — matrizes de confusão...")
fig1, axes = plt.subplots(1, 4, figsize=(18, 4))
fig1.suptitle("Matrizes de Confusão — todos os modelos", fontsize=14, fontweight="bold", y=1.02)
for ax, (name, m) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, m["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Pred: Ficou", "Pred: Churnou"],
        yticklabels=["Real: Ficou", "Real: Churnou"], cbar=False)
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"{name}\nTP={tp}  FP={fp}  FN={fn}  TN={tn}", fontsize=9)
    ax.set_xlabel("Predição")
    ax.set_ylabel("Realidade")
plt.tight_layout()
plt.savefig("fig1_confusion_matrices.png", dpi=150, bbox_inches="tight")
ok("fig1_confusion_matrices.png salva")

# ── FIGURA 2: Curvas ROC ──
print("  ⏳ Gerando fig2 — curvas ROC...")
fig2, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório (AUROC=0.50)")
for name, m in models.items():
    fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
    ax.plot(fpr, tpr, color=m["color"], lw=2, label=f"{name} (AUROC={m['auroc']:.3f})")
ax.set_title("Curvas ROC — comparação de modelos", fontsize=13, fontweight="bold")
ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("fig2_roc_curves.png", dpi=150, bbox_inches="tight")
ok("fig2_roc_curves.png salva")

# ── FIGURA 3: Curvas Precision-Recall ──
print("  ⏳ Gerando fig3 — curvas Precision-Recall...")
fig3, ax = plt.subplots(figsize=(8, 6))
baseline = y_test.mean()
ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Baseline aleatório (={baseline:.2f})")
for name, m in models.items():
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, m["y_proba"])
    ax.plot(rec_curve, prec_curve, color=m["color"], lw=2, label=f"{name} (AUPRC={m['auprc']:.3f})")
ax.set_title("Curvas Precision-Recall — comparação de modelos", fontsize=13, fontweight="bold")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("fig3_prc_curves.png", dpi=150, bbox_inches="tight")
ok("fig3_prc_curves.png salva")

# ── FIGURA 4: Barras de métricas ──
print("  ⏳ Gerando fig4 — comparação de métricas...")
metrics_names = ["AUROC", "AUPRC", "Precision", "Recall", "F1"]
fig4, axes4 = plt.subplots(1, 5, figsize=(18, 5))
fig4.suptitle("Comparação de métricas — todos os modelos", fontsize=14, fontweight="bold")
keys = ["auroc", "auprc", "prec", "rec", "f1"]
for ax, metric, key in zip(axes4, metrics_names, keys):
    names  = list(models.keys())
    values = [m[key] for m in models.values()]
    colors = [m["color"] for m in models.values()]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(metric, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (LDM proxy)", "\n(LDM proxy)") for n in names], fontsize=7, rotation=15, ha="right")
    ax.set_ylim([0, 1.1])
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig("fig4_metrics_comparison.png", dpi=150, bbox_inches="tight")
ok("fig4_metrics_comparison.png salva")

# ── FIGURA 5: Distribuição de scores ──
print("  ⏳ Gerando fig5 — distribuição de scores...")
fig5, axes5 = plt.subplots(2, 2, figsize=(12, 8))
fig5.suptitle("Distribuição de scores — churn vs. não-churn", fontsize=13, fontweight="bold")
for ax, (name, m) in zip(axes5.flatten(), models.items()):
    scores_0 = m["y_proba"][y_test == 0]
    scores_1 = m["y_proba"][y_test == 1]
    ax.hist(scores_0, bins=30, alpha=0.6, color="#4f8fff", label="Não churnou (0)", density=True)
    ax.hist(scores_1, bins=30, alpha=0.6, color="#e24b4a", label="Churnou (1)", density=True)
    ax.axvline(0.5, color="gray", linestyle="--", lw=1, label="Limiar=0.5")
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Score do modelo"); ax.set_ylabel("Densidade")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig5_score_distributions.png", dpi=150, bbox_inches="tight")
ok("fig5_score_distributions.png salva")

# ── RESUMO FINAL ──
print()
print("═" * 60)
print("  RESUMO FINAL")
print("═" * 60)
print(f"  {'Modelo':<30} {'AUROC':>7} {'AUPRC':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("  " + "─" * 58)
for name, m in models.items():
    label = name if tab_available or "Tab" not in name else name + " *"
    print(f"  {label:<30} {m['auroc']:>7.4f} {m['auprc']:>7.4f} {m['prec']:>7.4f} {m['rec']:>7.4f} {m['f1']:>7.4f}")

if not tab_available:
    print()
    print("  * TabTransformer não disponível. Instale: pip install pytorch-tabnet torch")

print()
print("  Arquivos gerados:")
for f in ["fig1_confusion_matrices.png","fig2_roc_curves.png","fig3_prc_curves.png",
          "fig4_metrics_comparison.png","fig5_score_distributions.png"]:
    print(f"    ✅ {f}")
print()
print("  🎉 Tudo pronto! Abra os arquivos PNG para ver os gráficos.")