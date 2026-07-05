"""
RELU — DUAS SITUAÇÕES COMPARADAS
==================================
Situação 1: relu na saída
  → o relu é aplicado diretamente na saída do modelo
  → problema: corta valores negativos durante o treinamento
  → pode atrapalhar a convergência em problemas de regressão

Situação 2: relu na camada oculta (arquitetura correta)
  → um neurônio oculto com relu processa a entrada
  → a saída final não tem relu (problema de regressão)
  → é assim que redes neurais reais funcionam
"""

# ============================================================
# DATASET (altura + idade → peso)
# ============================================================
dataset = [
    ([1.60, 20], 55.0), ([1.65, 25], 59.0), ([1.70, 30], 64.0),
    ([1.75, 22], 70.0), ([1.80, 35], 75.0), ([1.85, 28], 81.0),
    ([1.90, 40], 85.0), ([1.70, 45], 68.0), ([1.75, 50], 73.0),
    ([1.80, 55], 78.0),
]

n          = len(dataset)
n_features = 2

# normalização z-score
medias  = [sum(dataset[i][0][j] for i in range(n)) / n for j in range(n_features)]
desvios = [(sum((dataset[i][0][j] - medias[j]) ** 2 for i in range(n)) / n) ** 0.5 for j in range(n_features)]

def normalizar(X):
    return [(X[j] - medias[j]) / desvios[j] for j in range(n_features)]

def relu(x):
    return max(0.0, x)

def mse(previsoes, reais):
    return sum((p - r) ** 2 for p, r in zip(previsoes, reais)) / len(reais)


# ============================================================
# SITUAÇÃO 1 — relu na saída
# ============================================================
print("=" * 60)
print("SITUAÇÃO 1 — relu na saída")
print("=" * 60)
print("""
  arquitetura:
    entrada → produto escalar → + viés → relu → saída

  f(X) = relu(W · X + b)

  problema esperado:
    nas primeiras épocas, W e b ainda são ruins
    o produto escalar pode dar valores negativos
    relu corta tudo para zero → gradiente = 0 → modelo para de aprender
    isso é chamado de "neurônio morto" (dying relu)
""")

W1 = [1.0, 1.0]
b1 = 1.0
lr = 0.1

def modelo_v1(X_norm, W, b):
    result = sum(W[j] * X_norm[j] for j in range(n_features)) + b
    return relu(result)   # ← relu na saída

def gradientes_v1(W, b):
    dL_dW = [0.0] * n_features
    dL_db = 0.0
    for X, y in dataset:
        X_norm = normalizar(X)
        z = sum(W[j] * X_norm[j] for j in range(n_features)) + b
        y_prev = relu(z)
        e = y_prev - y
        # derivada do relu: 1 se z > 0, 0 se z <= 0
        drelu = 1.0 if z > 0 else 0.0
        for j in range(n_features):
            dL_dW[j] += 2 * e * drelu * X_norm[j]
        dL_db += 2 * e * drelu
    return [dL_dW[j] / n for j in range(n_features)], dL_db / n

print(f"{'Época':>6}  {'MSE':>10}  {'W[0]':>10}  {'W[1]':>10}  {'b':>10}  {'neurônios vivos':>16}")
print("-" * 70)

for epoca in range(500):
    previsoes = [modelo_v1(normalizar(X), W1, b1) for X, _ in dataset]
    L = mse(previsoes, [y for _, y in dataset])
    dL_dW, dL_db = gradientes_v1(W1, b1)

    # conta quantos exemplos têm z > 0 (neurônio "vivo")
    vivos = sum(1 for X, _ in dataset if sum(W1[j]*normalizar(X)[j] for j in range(n_features)) + b1 > 0)

    W1 = [W1[j] - lr * dL_dW[j] for j in range(n_features)]
    b1 = b1 - lr * dL_db

    if epoca % 50 == 0 or epoca == 499:
        print(f"{epoca:>6}  {L:>10.4f}  {W1[0]:>10.4f}  {W1[1]:>10.4f}  {b1:>10.4f}  {vivos:>10}/{n}")

previsoes_finais_v1 = [modelo_v1(normalizar(X), W1, b1) for X, _ in dataset]
print(f"\n  MSE final:  {mse(previsoes_finais_v1, [y for _,y in dataset]):.4f}")
print(f"  previsão para [1.75m, 30a]: {modelo_v1(normalizar([1.75, 30]), W1, b1):.1f} kg")


# ============================================================
# SITUAÇÃO 2 — relu na camada oculta (arquitetura correta)
# ============================================================
print("\n" + "=" * 60)
print("SITUAÇÃO 2 — relu na camada oculta")
print("=" * 60)
print("""
  arquitetura:
    entrada → [neurônio oculto com relu] → [neurônio de saída sem relu] → saída

  camada oculta (1 neurônio):  h = relu(W1 · X + b1)
  camada de saída:             y = W2 * h + b2

  o relu introduz não-linearidade na camada oculta
  a saída final não tem relu → pode prever qualquer valor real
""")

# parâmetros da camada oculta
W2 = [0.1, 0.1]   # pesos da entrada para o neurônio oculto
b2 = 0.1           # viés do neurônio oculto

# parâmetros da camada de saída
w3 = 0.1           # peso do neurônio oculto para a saída
b3 = 0.1           # viés da saída

lr = 0.01

def modelo_v2(X_norm, W2, b2, w3, b3):
    # camada oculta — com relu
    z_oculto = sum(W2[j] * X_norm[j] for j in range(n_features)) + b2
    h = relu(z_oculto)

    # camada de saída — sem relu
    y_prev = w3 * h + b3
    return y_prev, h, z_oculto

def gradientes_v2(W2, b2, w3, b3):
    dL_dW2 = [0.0] * n_features
    dL_db2 = 0.0
    dL_dw3 = 0.0
    dL_db3 = 0.0

    for X, y in dataset:
        X_norm = normalizar(X)
        y_prev, h, z_oculto = modelo_v2(X_norm, W2, b2, w3, b3)

        e = y_prev - y   # erro na saída

        # gradientes da camada de saída
        dL_dw3 += 2 * e * h
        dL_db3 += 2 * e

        # gradientes da camada oculta (regra da cadeia)
        drelu = 1.0 if z_oculto > 0 else 0.0
        for j in range(n_features):
            dL_dW2[j] += 2 * e * w3 * drelu * X_norm[j]
        dL_db2 += 2 * e * w3 * drelu

    return (
        [dL_dW2[j] / n for j in range(n_features)],
        dL_db2 / n,
        dL_dw3 / n,
        dL_db3 / n,
    )

print(f"{'Época':>6}  {'MSE':>10}  {'W2[0]':>10}  {'W2[1]':>10}  {'w3':>8}  {'b3':>8}")
print("-" * 60)

for epoca in range(500):
    previsoes = [modelo_v2(normalizar(X), W2, b2, w3, b3)[0] for X, _ in dataset]
    L = mse(previsoes, [y for _, y in dataset])
    dL_dW2, dL_db2, dL_dw3, dL_db3 = gradientes_v2(W2, b2, w3, b3)

    W2 = [W2[j] - lr * dL_dW2[j] for j in range(n_features)]
    b2 = b2 - lr * dL_db2
    w3 = w3 - lr * dL_dw3
    b3 = b3 - lr * dL_db3

    if epoca % 50 == 0 or epoca == 499:
        print(f"{epoca:>6}  {L:>10.4f}  {W2[0]:>10.4f}  {W2[1]:>10.4f}  {w3:>8.4f}  {b3:>8.4f}")

previsoes_finais_v2 = [modelo_v2(normalizar(X), W2, b2, w3, b3)[0] for X, _ in dataset]
print(f"\n  MSE final:  {mse(previsoes_finais_v2, [y for _,y in dataset]):.4f}")
print(f"  previsão para [1.75m, 30a]: {modelo_v2(normalizar([1.75, 30]), W2, b2, w3, b3)[0]:.1f} kg")


# ============================================================
# COMPARATIVO FINAL
# ============================================================
print("\n" + "=" * 60)
print("COMPARATIVO")
print("=" * 60)
print(f"  {'':35}  {'sit. 1':>10}  {'sit. 2':>10}")
print(f"  {'relu na saída':35}  {'sim':>10}  {'não':>10}")
print(f"  {'relu na camada oculta':35}  {'não':>10}  {'sim':>10}")
print(f"  {'risco de neurônio morto':35}  {'alto':>10}  {'baixo':>10}")
print(f"  {'MSE final':35}  {mse(previsoes_finais_v1, [y for _,y in dataset]):>10.4f}  {mse(previsoes_finais_v2, [y for _,y in dataset]):>10.4f}")
print(f"  {'previsão [1.75m, 30a]':35}  {modelo_v1(normalizar([1.75,30]), W1, b1):>10.1f}  {modelo_v2(normalizar([1.75,30]), W2, b2, w3, b3)[0]:>10.1f}")
print(f"  {'uso correto em redes neurais':35}  {'não':>10}  {'sim':>10}")