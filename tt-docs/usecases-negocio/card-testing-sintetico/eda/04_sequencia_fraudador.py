"""
4. Visualizar a sequência de transações de um cliente fraudador.
   Objetivo: mostrar visualmente o padrão micro -> micro -> golpe.
"""

import matplotlib.pyplot as plt
from utils import load_data, find_fraud_example, OUT_DIR


def main():
    df = load_data()
    cid = find_fraud_example(df)
    sub = df[df["clienteID"] == cid].sort_values("timestamp").reset_index(drop=True)

    print("=" * 60)
    print(f"CLIENTE FRAUDADOR SELECIONADO: {cid}")
    print("=" * 60)
    cols = ["timestamp", "tipo", "merchant", "valor", "saldo_antes", "saldo_depois", "isFraud"]
    print(sub[cols].to_string())

    print()
    print("Intervalo de tempo entre transações consecutivas:")
    print(sub["timestamp"].diff())

    # --- gráfico: valor da transação ao longo da sequência, destacando a fraude ---
    fig, ax = plt.subplots(figsize=(12, 6))

    cores = ["#C44E52" if f == 1 else "#4C72B0" for f in sub["isFraud"]]
    ax.bar(range(len(sub)), sub["valor"], color=cores)

    for i, row in sub.iterrows():
        ax.annotate(
            f"R$ {row['valor']:.2f}",
            (i, row["valor"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(
        [ts.strftime("%d/%m %H:%M") for ts in sub["timestamp"]],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Valor da transação (R$)")
    ax.set_xlabel("Transação (ordem cronológica)")
    ax.set_title(
        f"Sequência de transações do cliente fraudador {cid}\n"
        "Padrão card testing: micro-transações seguidas do golpe (barra vermelha)"
    )
    ax.axhline(10, color="gray", linestyle="--", linewidth=1, label="R$10 (linha de referência micro-transação)")
    ax.legend()

    plt.tight_layout()
    out_path = OUT_DIR / "04_sequencia_fraudador.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()
