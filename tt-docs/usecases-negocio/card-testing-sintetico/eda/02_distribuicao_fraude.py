"""
2. Distribuição de fraudes vs legítimas (nível de transação e nível de cliente)
"""

import matplotlib.pyplot as plt
from utils import load_data, OUT_DIR


def main():
    df = load_data()

    n_total = len(df)
    n_fraude = int(df["isFraud"].sum())
    n_legit = n_total - n_fraude
    pct_fraude = n_fraude / n_total * 100

    clientes_total = df["clienteID"].nunique()
    clientes_fraudadores = df.loc[df["isFraud"] == 1, "clienteID"].nunique()
    pct_clientes_fraudadores = clientes_fraudadores / clientes_total * 100

    print("=" * 60)
    print("NÍVEL DE TRANSAÇÃO")
    print("=" * 60)
    print(f"Total de transações: {n_total:,}".replace(",", "."))
    print(f"Legítimas (isFraud=0): {n_legit:,} ({100 - pct_fraude:.3f}%)".replace(",", "."))
    print(f"Fraudulentas (isFraud=1): {n_fraude:,} ({pct_fraude:.3f}%)".replace(",", "."))

    print()
    print("=" * 60)
    print("NÍVEL DE CLIENTE")
    print("=" * 60)
    print(f"Total de clientes: {clientes_total:,}".replace(",", "."))
    print(f"Clientes com >=1 fraude: {clientes_fraudadores:,} ({pct_clientes_fraudadores:.2f}%)".replace(",", "."))
    print(f"Clientes 100% legítimos: {clientes_total - clientes_fraudadores:,}".replace(",", "."))

    # cada cliente fraudador tem exatamente 1 transação marcada como fraude?
    fraudes_por_cliente = df[df["isFraud"] == 1].groupby("clienteID").size()
    print()
    print("Transações de fraude por cliente fraudador (describe):")
    print(fraudes_por_cliente.describe())

    # --- gráfico: contagem transações ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        ["Legítima", "Fraude"],
        [n_legit, n_fraude],
        color=["#4C72B0", "#C44E52"],
    )
    axes[0].set_yscale("log")
    axes[0].set_title(f"Transações: legítima vs fraude\n(n={n_total:,} | fraude={pct_fraude:.2f}%)".replace(",", "."))
    axes[0].set_ylabel("Contagem (escala log)")
    for i, v in enumerate([n_legit, n_fraude]):
        axes[0].text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom")

    axes[1].bar(
        ["Sem fraude", "Com fraude"],
        [clientes_total - clientes_fraudadores, clientes_fraudadores],
        color=["#4C72B0", "#C44E52"],
    )
    axes[1].set_title(
        f"Clientes: sem vs com fraude\n(n={clientes_total:,} | {pct_clientes_fraudadores:.2f}% dos clientes)".replace(",", ".")
    )
    axes[1].set_ylabel("Contagem de clientes")
    for i, v in enumerate([clientes_total - clientes_fraudadores, clientes_fraudadores]):
        axes[1].text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom")

    plt.tight_layout()
    out_path = OUT_DIR / "02_distribuicao_fraude.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()
