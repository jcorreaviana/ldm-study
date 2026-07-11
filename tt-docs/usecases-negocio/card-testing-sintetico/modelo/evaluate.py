"""
Avalia o transformer sequencial no conjunto de teste (clientes nunca vistos
no treino/validação).

Duas métricas, calculadas contra DOIS rótulos diferentes -- não confundir:

1. AUPRC contra isFraud REAL (coluna 'isfraud_real' vinda do batch, nunca
   tocada pelo treino) -- é a métrica principal, diretamente comparável ao
   PR-AUC=0.82 do baseline XGBoost, porque mede a mesma coisa que o baseline
   media: acertar a fraude de fato.

2. Recall preventivo contra 'label_preventivo' -- o alvo usado no TREINO
   (ver utils.propagate_label_preventivo e dataset.py). Aqui olhamos
   especificamente para as posições em que label_preventivo=1 MAS
   isfraud_real=0 -- ou seja, as micro-transações precursoras de um golpe,
   que o modelo deveria aprender a marcar mesmo sem elas serem, elas
   mesmas, a fraude. Esse é o número que responde "o modelo alerta ANTES
   do golpe?" -- no baseline XGBoost, e no treino antigo contra isFraud
   puro, esse número era 0%.

Uso:
    python evaluate.py
(requer ter rodado train.py antes — usa os artefatos em modelo_out/)
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import Config
from dataset import make_dataloader
from model import TransformerFraudDetector
from utils import load_data, transform, Preprocessors


@torch.no_grad()
def predict(model, loader, device):
    """
    Roda o modelo e devolve um DataFrame linha-a-linha (uma linha por
    transação real, sem padding), com os DOIS rótulos lado a lado:
    'isfraud_real' (fraude de fato) e 'label_preventivo' (alvo de treino,
    propagado para as transações precursoras do golpe).
    """
    model.eval()
    registros = []
    for batch in loader:
        numeric = batch["numeric"].to(device)
        tipo_idx = batch["tipo_idx"].to(device)
        merchant_idx = batch["merchant_idx"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        valid_mask = batch["valid_mask"]
        label_preventivo = batch["labels"]        # alvo de treino (propagado)
        isfraud_real = batch["isfraud_real"]       # fraude de fato (nunca propagada)

        logits = model(numeric, tipo_idx, merchant_idx, attn_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        label_preventivo_np = label_preventivo.numpy()
        isfraud_real_np = isfraud_real.numpy()
        valid_np = valid_mask.numpy()

        for b_idx, cid in enumerate(batch["clienteID"]):
            L = int(valid_np[b_idx].sum())
            timestamps = batch["timestamps"][b_idx]
            for t in range(L):
                registros.append(
                    {
                        "clienteID": cid,
                        "timestamp": timestamps[t],
                        "isfraud_real": int(isfraud_real_np[b_idx, t]),
                        "label_preventivo": int(label_preventivo_np[b_idx, t]),
                        "proba_fraude": float(probs[b_idx, t]),
                    }
                )
    return pd.DataFrame(registros)


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prep = Preprocessors.load(cfg.out_dir / "preprocessors.pkl")
    with open(cfg.out_dir / "model_config.json") as f:
        model_cfg = json.load(f)
    with open(cfg.out_dir / "test_client_ids.json") as f:
        test_ids = json.load(f)

    df = load_data(cfg)
    df_t = transform(df, prep)
    test_loader = make_dataloader(df_t, test_ids, cfg, shuffle=False)

    model = TransformerFraudDetector(
        n_tipo=model_cfg["n_tipo"],
        n_merchant=model_cfg["n_merchant"],
        n_numeric=model_cfg["n_numeric"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        cat_emb_dim=model_cfg["cat_emb_dim"],
        max_len=model_cfg["max_len"],
    ).to(device)
    model.load_state_dict(torch.load(cfg.out_dir / "model.pt", map_location=device))

    preds = predict(model, test_loader, device)
    # format='mixed' -- alguns timestamps do dataset vêm sem microsegundos
    # (ex.: gerados com segundos exatos), o que faz pd.to_datetime sem
    # format explícito levantar ValueError ao misturar formatos na mesma
    # coluna. 'mixed' deixa o pandas inferir o formato linha a linha.
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], format="mixed")

    # ======================================================================
    # 1. MÉTRICA PRINCIPAL — AUPRC contra isFraud REAL (comparável ao baseline)
    # ======================================================================
    auprc_real = average_precision_score(preds["isfraud_real"], preds["proba_fraude"])
    print("=" * 70)
    print("1) AUPRC contra isFraud REAL (comparável com baseline PR-AUC=0.82)")
    print("=" * 70)
    print(f"n transações: {len(preds):,} | fraudes reais: {int(preds['isfraud_real'].sum())}".replace(",", "."))
    print(f"AUPRC: {auprc_real:.4f}")

    print()
    print("Precision / Recall / F1 em diferentes thresholds (contra isFraud real):")
    thresholds_result = []
    for th in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        y_pred = (preds["proba_fraude"] >= th).astype(int)
        p = precision_score(preds["isfraud_real"], y_pred, zero_division=0)
        r = recall_score(preds["isfraud_real"], y_pred, zero_division=0)
        f1 = f1_score(preds["isfraud_real"], y_pred, zero_division=0)
        print(f"  threshold={th:.1f} | precision={p:.3f} | recall={r:.3f} | f1={f1:.3f}")
        thresholds_result.append({"threshold": th, "precision": p, "recall": r, "f1": f1})

    # ======================================================================
    # 2. RECALL PREVENTIVO — contra label_preventivo, só nas micro-transações
    #    (label_preventivo=1 mas isfraud_real=0 -- precursoras do golpe)
    # ======================================================================
    precursoras = preds[(preds["label_preventivo"] == 1) & (preds["isfraud_real"] == 0)]
    golpe = preds[preds["isfraud_real"] == 1]
    outras_legit = preds[(preds["label_preventivo"] == 0) & (preds["isfraud_real"] == 0)]

    print()
    print("=" * 70)
    print("2) RECALL PREVENTIVO — micro-transações antes do golpe (via label_preventivo)")
    print("=" * 70)
    print(f"Transações precursoras no teste (label_preventivo=1, isfraud_real=0): {len(precursoras)}")
    print(precursoras["proba_fraude"].describe())

    recall_preventivo = (
        (precursoras["proba_fraude"] >= cfg.alert_threshold).mean() if len(precursoras) else float("nan")
    )
    print(f"\nRecall preventivo (threshold={cfg.alert_threshold}): {recall_preventivo * 100:.1f}%")
    print("(no baseline XGBoost, e no treino antigo contra isFraud puro, esse número era 0.0%)")

    # recall por golpe (sequência): pelo menos 1 dos precursores foi alertado?
    precursoras_por_golpe = precursoras.groupby("clienteID")["proba_fraude"].apply(
        lambda s: (s >= cfg.alert_threshold).any()
    )
    recall_sequencial = precursoras_por_golpe.mean() if len(precursoras_por_golpe) else float("nan")
    print(
        f"Golpes com pelo menos 1 alerta preventivo antes do prejuízo: "
        f"{recall_sequencial * 100:.1f}% ({int(precursoras_por_golpe.sum())}/{len(precursoras_por_golpe)})"
    )

    print()
    print("Para comparação -- probabilidade prevista para o golpe em si (isfraud_real=1):")
    print(golpe["proba_fraude"].describe())

    # --- gráficos ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.hist(outras_legit["proba_fraude"].dropna(), bins=40, alpha=0.5, label="Legítima (label_preventivo=0)", color="#4C72B0", density=True)
    ax.hist(precursoras["proba_fraude"].dropna(), bins=40, alpha=0.7, label="Precursora (label_preventivo=1, isfraud_real=0)", color="#DD8452", density=True)
    ax.hist(golpe["proba_fraude"].dropna(), bins=40, alpha=0.7, label="Golpe (isfraud_real=1)", color="#C44E52", density=True)
    ax.axvline(cfg.alert_threshold, color="gray", linestyle="--", label=f"threshold={cfg.alert_threshold}")
    ax.set_xlabel("Probabilidade de fraude prevista pelo transformer")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição de probabilidade por tipo de transação\n(objetivo: precursora também subir acima do threshold)")
    ax.legend(fontsize=8)

    ax = axes[1]
    thresholds_df = pd.DataFrame(thresholds_result)
    ax.plot(thresholds_df["threshold"], thresholds_df["precision"], marker="o", label="Precision (isFraud real)")
    ax.plot(thresholds_df["threshold"], thresholds_df["recall"], marker="o", label="Recall (isFraud real)")
    ax.plot(thresholds_df["threshold"], thresholds_df["f1"], marker="o", label="F1 (isFraud real)")
    ax.set_xlabel("Threshold de decisão")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs threshold (contra isFraud real)")
    ax.legend()

    plt.tight_layout()
    out_path = cfg.out_dir / "avaliacao_transformer.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {out_path}")

    metrics_out = {
        "auprc_isfraud_real": float(auprc_real),
        "n_teste": int(len(preds)),
        "n_fraudes_reais_teste": int(preds["isfraud_real"].sum()),
        "thresholds_isfraud_real": thresholds_result,
        "recall_preventivo_micro_transacional": float(recall_preventivo),
        "recall_preventivo_sequencial_alerta_antes_golpe": float(recall_sequencial),
        "n_precursoras_teste": int(len(precursoras)),
        "n_golpes_teste": int(len(precursoras_por_golpe)),
        "alert_threshold": cfg.alert_threshold,
    }
    with open(cfg.out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Métricas salvas em: {cfg.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
