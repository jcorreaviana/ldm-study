"""
1. Shape e tipos de coluna do dataset_sintetico.csv
"""

from utils import load_data


def main():
    df = load_data()

    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(f"Linhas: {df.shape[0]:,}".replace(",", "."))
    print(f"Colunas: {df.shape[1]}")

    print()
    print("=" * 60)
    print("TIPOS DE COLUNA (dtypes)")
    print("=" * 60)
    print(df.dtypes)

    print()
    print("=" * 60)
    print("NULOS POR COLUNA")
    print("=" * 60)
    print(df.isnull().sum())

    print()
    print("=" * 60)
    print("CARDINALIDADE")
    print("=" * 60)
    print(f"clienteID únicos: {df['clienteID'].nunique():,}".replace(",", "."))
    print(f"merchant únicos: {df['merchant'].nunique()}")
    print(f"tipo únicos: {df['tipo'].unique().tolist()}")
    print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")

    print()
    print("=" * 60)
    print("AMOSTRA (5 primeiras linhas)")
    print("=" * 60)
    print(df.head().to_string())


if __name__ == "__main__":
    main()
