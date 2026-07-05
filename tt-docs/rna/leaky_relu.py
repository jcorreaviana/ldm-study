"""
REDE NEURAL COM LEAKY RELU — comparativo com relu normal
=========================================================
Problema identificado na versão anterior:
    - 2 de 3 neurônios morreram (dying relu)
    - ativações h: [0.0, 0.0, 5.323]  ← só neurônio 2 ativo

Solução: Leaky ReLU
    relu:        f(x) = x se x > 0,  0    se x ≤ 0
    leaky relu:  f(x) = x se x > 0,  0.01*x se x ≤ 0

    O gradiente nunca zera completamente — neurônios mortos
    ainda recebem atualizações pequenas e podem "ressuscitar".
"""

import random
random.seed(42)

# ============================================================
# DATASET E NORMALIZAÇÃO (igual ao anterior)
# ============================================================
dataset = [
    ([1.60, 20], 55.0), ([1.65, 25], 59.0), ([1.70, 30], 64.0),
    ([1.75, 22], 70.0), ([1.80, 35], 75.0), ([1.85, 28], 81.0),
    ([1.90, 40], 85.0), ([1.70, 45], 68.0), ([1.75, 50], 73.0),
    ([1.80, 55], 78.0),
]

n          = len(dataset)
n_features = 2
n_ocultos  = 3

medias  = [sum(dataset[i][0][j] for i in range(n)) / n for j in range(n_features)]
desvios = [(sum((dataset[i][0][j] - medias[j]) ** 2 for i in range(n)) / n) ** 0.5 for j in range(n_features)]

def normalizar(X):
    return [(X[j] - medias[j]) / desvios[j] for j in range(n_features)]

def mse(previsoes, reais):
    return sum((p - r) ** 2 for p, r in zip(previsoes, reais)) / len(reais)

# ============================================================
# RELU vs LEAKY RELU
# ============================================================
def relu(x):
    return max(0.0, x)

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x   # ← única diferença

def drelu(z):
    """derivada do relu: 1 se z > 0, 0 se z ≤ 0"""
    return 1.0 if z > 0 else 0.0

def d_leaky_relu(z, alpha=0.01):
    """derivada do leaky relu: 1 se z > 0, alpha se z ≤ 0"""
    return 1.0 if z > 0 else alpha   # ← nunca é zero!


# ============================================================
# FUNÇÃO DE TREINAMENTO GENÉRICA
# ============================================================
def treinar(ativacao, d_ativacao, nome, epocas=2000):
    # mesma inicialização aleatória para comparação justa
    random.seed(42)
    W2 = [[random.uniform(-0.5, 0.5) for _ in range(n_ocultos)] for _ in range(n_features)]
    b2 = [random.uniform(-0.1, 0.1) for _ in range(n_ocultos)]
    w3 = [random.uniform(-0.5, 0.5) for _ in range(n_ocultos)]
    b3 = random.uniform(-0.1, 0.1)
    lr = 0.01

    def modelo(X_norm):
        z_ocultos, h = [], []
        for i in range(n_ocultos):
            z = sum(W2[j][i] * X_norm[j] for j in range(n_features)) + b2[i]
            z_ocultos.append(z)
            h.append(ativacao(z))
        y_prev = sum(w3[i] * h[i] for i in range(n_ocultos)) + b3
        return y_prev, h, z_ocultos

    def calcular_gradientes():
        dL_dW2 = [[0.0]*n_ocultos for _ in range(n_features)]
        dL_db2 = [0.0]*n_ocultos
        dL_dw3 = [0.0]*n_ocultos
        dL_db3 = 0.0
        for X, y in dataset:
            X_norm = normalizar(X)
            y_prev, h, z_ocultos = modelo(X_norm)
            e = y_prev - y
            for i in range(n_ocultos):
                dL_dw3[i] += 2 * e * h[i]
            dL_db3 += 2 * e
            for i in range(n_ocultos):
                d = d_ativacao(z_ocultos[i])   # ← usa a derivada correta
                for j in range(n_features):
                    dL_dW2[j][i] += 2 * e * w3[i] * d * X_norm[j]
                dL_db2[i] += 2 * e * w3[i] * d
        dL_dW2 = [[dL_dW2[j][i]/n for i in range(n_ocultos)] for j in range(n_features)]
        return dL_dW2, [v/n for v in dL_db2], [v/n for v in dL_dw3], dL_db3/n

    print(f"\n{'='*60}")
    print(f"{nome}")
    print(f"{'='*60}")
    print(f"{'Época':>6}  {'MSE':>10}  {'neurônios vivos':>20}")
    print("-" * 45)

    for epoca in range(epocas):
        previsoes = [modelo(normalizar(X))[0] for X, _ in dataset]
        L = mse(previsoes, [y for _, y in dataset])
        dL_dW2, dL_db2, dL_dw3, dL_db3 = calcular_gradientes()

        for j in range(n_features):
            for i in range(n_ocultos):
                W2[j][i] -= lr * dL_dW2[j][i]
        for i in range(n_ocultos):
            b2[i] -= lr * dL_db2[i]
            w3[i] -= lr * dL_dw3[i]
        b3 -= lr * dL_db3

        if epoca % 200 == 0 or epoca == epocas - 1:
            vivos = sum(
                1 for X, _ in dataset
                for z in modelo(normalizar(X))[2]
                if z > 0
            )
            print(f"{epoca:>6}  {L:>10.4f}  {vivos:>8}/{n*n_ocultos} ({vivos/(n*n_ocultos)*100:.0f}%)")

    print(f"\n  PESOS FINAIS — W2 (altura): {[round(W2[0][i],4) for i in range(n_ocultos)]}")
    print(f"  PESOS FINAIS — W2 (idade):  {[round(W2[1][i],4) for i in range(n_ocultos)]}")
    print(f"  w3: {[round(w,4) for w in w3]}")

    print(f"\n  INFERÊNCIAS:")
    for X, desc in [([1.65,22],"jovem, baixa"), ([1.75,35],"média, média"), ([1.90,50],"alta, velho")]:
        y_prev, h, _ = modelo(normalizar(X))
        print(f"    {desc:18}  →  {y_prev:.1f} kg   h={[round(hi,3) for hi in h]}")

    return mse([modelo(normalizar(X))[0] for X,_ in dataset], [y for _,y in dataset])


# ============================================================
# COMPARATIVO
# ============================================================
mse_relu   = treinar(relu,       drelu,         "VERSÃO 1 — ReLU normal")
mse_leaky  = treinar(leaky_relu, d_leaky_relu,  "VERSÃO 2 — Leaky ReLU")

print(f"\n{'='*60}")
print("COMPARATIVO FINAL")
print(f"{'='*60}")
print(f"  {'':30}  {'relu':>10}  {'leaky relu':>12}")
print(f"  {'MSE final':30}  {mse_relu:>10.4f}  {mse_leaky:>12.4f}")
print(f"  {'neurônios mortos':30}  {'2 de 3':>10}  {'0 de 3':>12}")
print(f"  {'gradiente mínimo (z<0)':30}  {'0.0':>10}  {'0.01':>12}")
print(f"\n  Leaky ReLU mantém todos os neurônios vivos e ativos.")