"""
6. Distribuição de valores: fraude vs legítimo
"""

import matplotlib.pyplot as plt
from utils import load_data, OUT_DIR


def main():
    df = load_data()

    legit = df.loc[df["isFraud"] == 0, "valor"]
    fraude = df.loc[df["isFraud"] == 1, "valor"]

    print("=" * 60)
    print("VALOR — LEGÍTIMAS (isFraud=0)")
    print("=" * 60)
    print(legit.describe())

    print()
    print("=" * 60)
    print("VALOR — FRAUDES (isFraud=1)")
    print("=" * 60)
    print(fraude.describe())

    print()
    print(f"Valor médio fraude é {fraude.mean() / legit.mean():.2f}x o valor médio legítimo")

    # --- gráfico: histogramas + boxplot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bins = 60
    axes[0].hist(legit, bins=bins, alpha=0.6, label="Legítima", color="#4C72B0", density=True)
    axes[0].hist(fraude, bins=bins, alpha=0.6, label="Fraude", color="#C44E52", density=True)
    axes[0].set_title("Distribuição de valor: fraude vs legítima")
    axes[0].set_xlabel("Valor (R$)")
    axes[0].set_ylabel("Densidade")
    axes[0].legend()

    axes[1].boxplot(
        [legit, fraude],
        tick_labels=["Legítima", "Fraude"],
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0"),
    )
    axes[1].set_title("Boxplot de valor: fraude vs legítima")
    axes[1].set_ylabel("Valor (R$)")

    plt.tight_layout()
    out_path = OUT_DIR / "06_distribuicao_valores.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()
