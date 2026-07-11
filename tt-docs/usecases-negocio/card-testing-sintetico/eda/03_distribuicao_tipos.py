"""
3. Distribuição de tipos de transação (geral e fraude vs legítima)
"""

import matplotlib.pyplot as plt
from utils import load_data, OUT_DIR


def main():
    df = load_data()

    print("=" * 60)
    print("DISTRIBUIÇÃO GERAL DE TIPOS")
    print("=" * 60)
    print(df["tipo"].value_counts())
    print()
    print(df["tipo"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

    print()
    print("=" * 60)
    print("TIPOS ENTRE TRANSAÇÕES FRAUDULENTAS (isFraud=1)")
    print("=" * 60)
    print(df.loc[df["isFraud"] == 1, "tipo"].value_counts())

    print()
    print("=" * 60)
    print("MERCHANT ENTRE TRANSAÇÕES FRAUDULENTAS (isFraud=1)")
    print("=" * 60)
    print(df.loc[df["isFraud"] == 1, "merchant"].value_counts())
    print(
        "\nObs.: a transação marcada isFraud=1 é sempre o 'golpe' final. "
        "As micro-transações de teste que a precedem aparecem como isFraud=0, "
        "isso é justamente o motivo pelo qual o modelo atual (transação a "
        "transação) não enxerga o padrão."
    )

    # --- gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts_geral = df["tipo"].value_counts()
    axes[0].bar(counts_geral.index, counts_geral.values, color="#4C72B0")
    axes[0].set_title("Distribuição geral de tipos de transação")
    axes[0].set_ylabel("Contagem")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(counts_geral.values):
        axes[0].text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom")

    counts_fraude = df.loc[df["isFraud"] == 1, "tipo"].value_counts()
    axes[1].bar(counts_fraude.index, counts_fraude.values, color="#C44E52")
    axes[1].set_title("Tipos de transação — apenas fraudes (isFraud=1)")
    axes[1].set_ylabel("Contagem")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(counts_fraude.values):
        axes[1].text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom")

    plt.tight_layout()
    out_path = OUT_DIR / "03_distribuicao_tipos.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()
