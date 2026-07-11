"""
Roda o pipeline completo do transformer: treino -> avaliação.
Uso: python run_all.py
Requer: pip install -r requirements.txt
"""

import train
import evaluate


def main():
    print("\n" + "#" * 70)
    print("# train")
    print("#" * 70 + "\n")
    train.main()

    print("\n" + "#" * 70)
    print("# evaluate")
    print("#" * 70 + "\n")
    evaluate.main()


if __name__ == "__main__":
    main()
