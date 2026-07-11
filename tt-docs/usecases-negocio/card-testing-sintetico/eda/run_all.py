"""
Roda todos os scripts de EDA em sequência.
Uso: python run_all.py
"""

import importlib

SCRIPTS = [
    "01_shape_dtypes",
    "02_distribuicao_fraude",
    "03_distribuicao_tipos",
    "04_sequencia_fraudador",
    "05_sequencia_legitimo",
    "06_distribuicao_valores",
    "07_distribuicao_horarios",
]


def main():
    for nome in SCRIPTS:
        print("\n" + "#" * 70)
        print(f"# {nome}")
        print("#" * 70 + "\n")
        mod = importlib.import_module(nome)
        mod.main()


if __name__ == "__main__":
    main()
