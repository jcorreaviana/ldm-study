"""
EDA - PaySim (mobile money, 6.3M transacoes)
Python 3.12 / pandas / matplotlib.

O dataset.csv fica ao lado deste projeto mas NAO deve ser versionado -
so os scripts. Aponte --path para onde voce guarda o arquivo.

Cobre os itens 1-6 do EDA_paysim.md (shape, tipos, desbalanceamento, fraude
por tipo, sequencia de drenagem, limite de R$10.000). Para a analise de
nameDest / event stream (item 7), use analisar_namedest.py.

Rode: python paysim_eda.py --path ../dataset.csv
Gera um PNG por vez em ./eda_out/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path(__file__).parent / "eda_out"


def load(path: str) -> pd.DataFrame:
    dtypes = {
        "step": "int32",
        "type": "category",
        "amount": "float64",
        "nameOrig": "string",
        "oldbalanceOrg": "float64",
        "newbalanceOrig": "float64",
        "nameDest": "string",
        "oldbalanceDest": "float64",
        "newbalanceDest": "float64",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    return pd.read_csv(path, dtype=dtypes)


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"salvo: {path}")
    plt.close(fig)


def eda_shape(df: pd.DataFrame):
    print("\n=== 1. Shape e tipos ===")
    print("shape:", df.shape)
    print(df.dtypes)


def eda_tx_types(df: pd.DataFrame):
    print("\n=== 2. Distribuicao por tipo ===")
    counts = df["type"].value_counts()
    print(pd.DataFrame({"count": counts, "pct": (counts / len(df) * 100).round(2)}))

    fig, ax = plt.subplots(figsize=(7, 4))
    counts.sort_values().plot(kind="barh", ax=ax, color="#378ADD")
    ax.set_xlabel("numero de transacoes")
    ax.set_title("Distribuicao dos tipos de transacao")
    save(fig, "01_tipos_transacao.png")


def eda_imbalance(df: pd.DataFrame):
    print("\n=== 3. Fraude vs legitima ===")
    counts = df["isFraud"].value_counts()
    print(counts, f"\ntaxa de fraude: {counts.get(1,0)/len(df)*100:.4f}%")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Legitima", "Fraude"], [counts.get(0, 0), counts.get(1, 0)],
           color=["#639922", "#E24B4A"], log=True)
    ax.set_ylabel("numero de transacoes (escala log)")
    save(fig, "02_desbalanceamento.png")


def eda_fraud_by_type(df: pd.DataFrame):
    print("\n=== 4. Fraude por tipo ===")
    tab = pd.crosstab(df["type"], df["isFraud"])
    print(tab)

    fraud_counts = df[df["isFraud"] == 1]["type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    fraud_counts.plot(kind="bar", ax=ax, color="#E24B4A")
    ax.set_ylabel("numero de fraudes")
    plt.xticks(rotation=0)
    save(fig, "03_fraude_por_tipo.png")


def eda_fraud_sequence(df: pd.DataFrame, n_examples: int = 3):
    print("\n=== 5. Sequencia de drenagem (exemplo) ===")
    fraud = df[df["isFraud"] == 1].sort_values(["step"]).reset_index(drop=True)
    cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest"]
    for i in range(0, n_examples * 2, 2):
        print(fraud.loc[i:i + 1, cols].to_string(index=False))
        print("-" * 60)


def eda_card_testing(df: pd.DataFrame, limite: float = 10_000.0):
    print("\n=== 6. Limite de R$10.000 vs fraude ===")
    tx = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    fraud_tx = tx[tx["isFraud"] == 1]
    abaixo = (fraud_tx["amount"] < limite).sum()
    acima = (fraud_tx["amount"] >= limite).sum()
    print(f"fraude < limite: {abaixo} ({abaixo/len(fraud_tx)*100:.2f}%)")
    print(f"fraude >= limite: {acima} ({acima/len(fraud_tx)*100:.2f}%)")

    flagged_hits = df[(df["isFlaggedFraud"] == 1) & (df["isFraud"] == 1)].shape[0]
    print(f"isFlaggedFraud recall: {flagged_hits/len(fraud_tx)*100:.2f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Fraude < limite", "Fraude >= limite"], [abaixo, acima],
           color=["#EDA100", "#E24B4A"])
    ax.set_ylabel("numero de transacoes fraudulentas")
    save(fig, "04_fraude_vs_limite.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    df = load(args.path)
    eda_shape(df)
    eda_tx_types(df)
    eda_imbalance(df)
    eda_fraud_by_type(df)
    eda_fraud_sequence(df)
    eda_card_testing(df)


if __name__ == "__main__":
    main()
