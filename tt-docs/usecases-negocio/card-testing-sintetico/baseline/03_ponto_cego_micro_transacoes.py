"""
3. Prova o ponto cego do baseline: qual probabilidade o modelo dá para as
   micro-transações de teste de cartão (que precedem o golpe) vs. para o
   próprio golpe?

Isso é o número que importa para o negócio: métricas agregadas (PR-AUC,
ROC-AUC, recall) parecem boas porque o modelo pega o golpe (transação
grande, de madrugada — sinal forte mesmo isolado). Mas o golpe só é
identificado DEPOIS que o dinheiro já saiu. O objetivo de negócio é
identificar o card testing ANTES do golpe, e é exatamente aí que o
baseline falha: as micro-transações recebem probabilidade próxima de zero.
"""

import pickle

import matplotlib.pyplot as plt
import pandas as pd

import utils
from utils import OUT_DIR


def main():
    with open(OUT_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    df = utils.load_data()
    X, y, meta = utils.build_features(df)
    df = df.copy()
    df["proba_fraude"] = model.predict_proba(X)[:, 1]

    fraud_clients = df.loc[df["isFraud"] == 1, "clienteID"].unique()

    sequencias = []
    for cid in fraud_clients:
        grp = df[df["clienteID"] == cid].sort_values("timestamp").reset_index(drop=True)
        posicoes_golpe = grp.index[grp["isFraud"] == 1].tolist()
        for pos in posicoes_golpe:
            if pos >= 2:
                anteriores = grp.iloc[pos - 2:pos]
                if (anteriores["valor"] < 10).all():
                    sequencias.append(grp.iloc[pos - 2:pos + 1])

    seq_df = pd.concat(sequencias)
    micro = seq_df[seq_df["isFraud"] == 0]
    golpe = seq_df[seq_df["isFraud"] == 1]

    print("=" * 60)
    print(f"Sequências micro->micro->golpe encontradas: {len(sequencias)}")
    print("=" * 60)

    print()
    print("Probabilidade prevista pelo baseline — MICRO-transações (isFraud=0, deveriam soar suspeitas):")
    print(micro["proba_fraude"].describe())

    print()
    print("Probabilidade prevista pelo baseline — GOLPE (isFraud=1):")
    print(golpe["proba_fraude"].describe())

    pct_micro_abaixo_50 = (micro["proba_fraude"] < 0.5).mean() * 100
    print()
    print(f"% de micro-transações com probabilidade < 0.5 (não alertadas): {pct_micro_abaixo_50:.1f}%")

    # --- gráfico ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(micro["proba_fraude"], bins=40, alpha=0.7, label="Micro-transações (isFraud=0)", color="#4C72B0")
    ax.hist(golpe["proba_fraude"], bins=40, alpha=0.7, label="Golpe (isFraud=1)", color="#C44E52")
    ax.axvline(0.5, color="gray", linestyle="--", label="threshold=0.5")
    ax.set_xlabel("Probabilidade de fraude prevista pelo baseline")
    ax.set_ylabel("Contagem")
    ax.set_title(
        "Ponto cego do baseline: micro-transações de teste ficam quase em 0,\n"
        "só o golpe final é identificado — tarde demais para prevenir a perda"
    )
    ax.legend()

    plt.tight_layout()
    out_path = OUT_DIR / "03_ponto_cego_micro_transacoes.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()
