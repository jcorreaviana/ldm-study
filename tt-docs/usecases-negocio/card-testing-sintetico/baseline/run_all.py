"""
Roda o pipeline completo do baseline: treino -> avaliação.
Uso: python run_all.py
"""

import importlib

SCRIPTS = [
    "01_train_baseline",
    "02_avaliar_baseline",
    "03_ponto_cego_micro_transacoes",
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
