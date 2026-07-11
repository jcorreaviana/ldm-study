"""
7. Distribuição de horários: fraude vs legítimo
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, OUT_DIR


def main():
    df = load_data()

    legit = df.loc[df["isFraud"] == 0, "hora"]
    fraude = df.loc[df["isFraud"] == 1, "hora"]

    print("=" * 60)
    print("HORA DO DIA — LEGÍTIMAS (isFraud=0)")
    print("=" * 60)
    print(legit.describe())

    print()
    print("=" * 60)
    print("HORA DO DIA — FRAUDES (isFraud=1)")
    print("=" * 60)
    print(fraude.describe())

    print()
    print("Contagem de fraudes por hora:")
    print(fraude.value_counts().sort_index())

    # --- gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bins = np.arange(0, 25) - 0.5
    axes[0].hist(legit, bins=bins, alpha=0.6, label="Legítima", color="#4C72B0", density=True)
    axes[0].hist(fraude, bins=bins, alpha=0.6, label="Fraude", color="#C44E52", density=True)
    axes[0].set_title("Distribuição de horário: fraude vs legítima")
    axes[0].set_xlabel("Hora do dia (0-23)")
    axes[0].set_ylabel("Densidade")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()

    axes[1].boxplot(
        [legit, fraude],
        tick_labels=["Legítima", "Fraude"],
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0"),
    )
    axes[1].set_title("Boxplot de horário: fraude vs legítima")
    axes[1].set_ylabel("Hora do dia")

    plt.tight_layout()
    out_path = OUT_DIR / "07_distribuicao_horarios.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()
