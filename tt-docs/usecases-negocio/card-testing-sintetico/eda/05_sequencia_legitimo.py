"""
5. Visualizar a sequência de transações de um cliente legítimo.
   Objetivo: confirmar a ausência do padrão micro -> micro -> golpe.
"""

import matplotlib.pyplot as plt
from utils import load_data, find_legit_example, OUT_DIR


def main():
    df = load_data()
    cid = find_legit_example(df)
    sub = df[df["clienteID"] == cid].sort_values("timestamp").reset_index(drop=True)

    print("=" * 60)
    print(f"CLIENTE LEGÍTIMO SELECIONADO: {cid}")
    print("=" * 60)
    cols = ["timestamp", "tipo", "merchant", "valor", "saldo_antes", "saldo_depois", "isFraud"]
    print(sub[cols].to_string())

    print()
    print(f"Valor mínimo: R$ {sub['valor'].min():.2f} | Valor máximo: R$ {sub['valor'].max():.2f}")
    n_micro = int((sub["valor"] < 10).sum())
    print(f"Transações abaixo de R$10 (micro): {n_micro} de {len(sub)}")

    # --- gráfico: valor da transação ao longo da sequência ---
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(range(len(sub)), sub["valor"], color="#4C72B0")

    for i, row in sub.iterrows():
        ax.annotate(
            f"R$ {row['valor']:.2f}",
            (i, row["valor"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
        )

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(
        [ts.strftime("%d/%m %H:%M") for ts in sub["timestamp"]],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Valor da transação (R$)")
    ax.set_xlabel("Transação (ordem cronológica)")
    ax.set_title(
        f"Sequência de transações do cliente legítimo {cid}\n"
        "Sem padrão de micro-transações seguidas de golpe"
    )
    ax.axhline(10, color="gray", linestyle="--", linewidth=1, label="R$10 (linha de referência micro-transação)")
    ax.legend()

    plt.tight_layout()
    out_path = OUT_DIR / "05_sequencia_legitimo.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()
