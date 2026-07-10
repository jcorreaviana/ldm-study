"""
Analise exaustiva de nameDest como ancora do event stream (item 7 do
EDA_paysim.md). Substitui as estimativas por amostragem feitas quando o
sandbox de execucao estava indisponivel.

Responde:
  1. quantas contas nameDest tem mais de 1 transacao? distribuicao?
  2. tamanho medio (e mediana) da sequencia por nameDest
  3. contas nameDest associadas a fraude tem mais transacoes que as legitimas?
  4. sugestao de comprimento de janela de contexto
  5. taxa de contas com label misto (recebem legitimas E fraude)

Rode: python analisar_namedest.py --path ../dataset.csv
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--limite", type=float, default=10_000.0)
    args = parser.parse_args()

    dtypes = {
        "step": "int32", "type": "category", "amount": "float64",
        "nameOrig": "string", "nameDest": "string", "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    usecols = list(dtypes.keys())
    df = pd.read_csv(args.path, dtype=dtypes, usecols=usecols)

    tx = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]

    # 1 e 2: distribuicao de transacoes por nameDest
    counts = tx.groupby("nameDest").size()
    print("=== contas nameDest (TRANSFER/CASH_OUT) ===")
    print(f"total de contas unicas: {counts.shape[0]:,}")
    print(f"com >1 transacao: {(counts > 1).sum():,} "
          f"({(counts > 1).mean()*100:.2f}%)")
    print(counts.describe())
    print("percentis:", counts.quantile([.5, .75, .9, .95, .99]).to_dict())

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.clip(upper=counts.quantile(0.99)).plot(kind="hist", bins=50, ax=ax, color="#378ADD")
    ax.set_xlabel("transacoes recebidas por conta destino (p99 truncado)")
    ax.set_title("Distribuicao do tamanho da sequencia por nameDest")
    fig.savefig("eda_out/05_distribuicao_sequencia_namedest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3: fraude vs legitima
    dest_fraud = tx.groupby("nameDest")["isFraud"].max()
    contas_com_fraude = dest_fraud[dest_fraud == 1].index
    contas_sem_fraude = dest_fraud[dest_fraud == 0].index

    print("\n=== volume de transacoes: contas com fraude vs sem fraude ===")
    print("com fraude - media:", counts.loc[contas_com_fraude].mean(),
          "mediana:", counts.loc[contas_com_fraude].median())
    print("sem fraude - media:", counts.loc[contas_sem_fraude].mean(),
          "mediana:", counts.loc[contas_sem_fraude].median())

    # 5: contas com label misto (legitima + fraude)
    misto = tx.groupby("nameDest")["isFraud"].nunique()
    n_misto = (misto == 2).sum()
    print(f"\ncontas que recebem transacao legitima E fraude: {n_misto:,} "
          f"({n_misto/misto.shape[0]*100:.4f}% das contas)")

    # 4: sugestao de janela de contexto
    mediana = counts.median()
    p90 = counts.quantile(0.9)
    print(f"\nsugestao de janela de contexto: mediana={mediana:.0f}, "
          f"p90={p90:.0f} -> comece testando com {int(p90)} (cobre 90% das contas "
          f"sem truncar, com padding para o resto)")


if __name__ == "__main__":
    main()
