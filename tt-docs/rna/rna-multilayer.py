"""
REDE NEURAL COM MÚLTIPLAS CAMADAS — versão estável
====================================================
Correções para redes profundas (5+ camadas):

    1. Inicialização He
       → pesos calibrados pela profundidade
       → evita explosão/desaparecimento de gradiente no início

    2. Learning rate menor (0.001)
       → passos menores = mais estável em redes profundas

    3. Gradient clipping
       → limita o tamanho máximo do gradiente
       → evita que um passo muito grande desestabilize o treino

Mude n_camadas para testar 1, 2, 3, 5 camadas.
"""

import random
import math
random.seed(42)

# ============================================================
# CONFIGURAÇÃO — mude aqui para testar diferentes arquiteturas
# ============================================================
n_camadas  = 100  # ← experimente 1, 2, 3, 5
n_ocultos  = 3
lr         = 0.001    # menor que antes (0.01 → 0.001)
max_grad   = 1.0      # gradient clipping
epocas     = 3000

# ============================================================
# DATASET E NORMALIZAÇÃO
# ============================================================
dataset = [
    ([1.60, 20], 55.0), ([1.65, 25], 59.0), ([1.70, 30], 64.0),
    ([1.75, 22], 70.0), ([1.80, 35], 75.0), ([1.85, 28], 81.0),
    ([1.90, 40], 85.0), ([1.70, 45], 68.0), ([1.75, 50], 73.0),
    ([1.80, 55], 78.0),
]

n          = len(dataset)
n_features = 2

medias  = [sum(dataset[i][0][j] for i in range(n)) / n for j in range(n_features)]
desvios = [(sum((dataset[i][0][j] - medias[j]) ** 2 for i in range(n)) / n) ** 0.5 for j in range(n_features)]

def normalizar(X):
    return [(X[j] - medias[j]) / desvios[j] for j in range(n_features)]

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def d_leaky_relu(z, alpha=0.01):
    return 1.0 if z > 0 else alpha

def mse(previsoes, reais):
    return sum((p - r) ** 2 for p, r in zip(previsoes, reais)) / len(reais)

def clip(valor, limite):
    """gradient clipping — limita o valor entre -limite e +limite"""
    return max(-limite, min(limite, valor))

# ============================================================
# PARÂMETROS — inicialização He
# ============================================================
# He initialization: pesos ~ N(0, sqrt(2/n_entradas))
# Por quê? Com leaky relu, a variância dos gradientes se mantém
# estável através das camadas — evita explosão/desaparecimento

def init_He(n_entradas, n_saidas):
    escala = math.sqrt(2.0 / n_entradas)
    return [[random.gauss(0, escala) for _ in range(n_saidas)]
            for _ in range(n_entradas)]

camadas_W = []
camadas_b = []

for l in range(n_camadas):
    n_entradas = n_features if l == 0 else n_ocultos
    camadas_W.append(init_He(n_entradas, n_ocultos))
    camadas_b.append([0.0] * n_ocultos)   # vieses iniciam em zero

w_saida = [random.gauss(0, math.sqrt(2.0 / n_ocultos)) for _ in range(n_ocultos)]
b_saida = 0.0

# conta parâmetros
total_params = (
    n_features * n_ocultos + n_ocultos +
    (n_camadas - 1) * (n_ocultos * n_ocultos + n_ocultos) +
    n_ocultos + 1
)

print("=" * 60)
print("ARQUITETURA")
print("=" * 60)
print(f"  features de entrada:  {n_features}")
print(f"  camadas ocultas:      {n_camadas}")
print(f"  neurônios por camada: {n_ocultos}")
print(f"  total de parâmetros:  {total_params}")
print(f"  ativação:             leaky relu (α=0.01)")
print(f"  inicialização:        He")
print(f"  learning rate:        {lr}")
print(f"  gradient clipping:    {max_grad}")

# ============================================================
# FORWARD PASS
# ============================================================
def forward(X_norm):
    todas_h = []
    todos_z = []
    entrada_atual = X_norm

    for l in range(n_camadas):
        h_camada, z_camada = [], []
        n_entradas = len(entrada_atual)
        for i in range(n_ocultos):
            z = sum(camadas_W[l][j][i] * entrada_atual[j]
                    for j in range(n_entradas)) + camadas_b[l][i]
            z_camada.append(z)
            h_camada.append(leaky_relu(z))
        todas_h.append(h_camada)
        todos_z.append(z_camada)
        entrada_atual = h_camada

    y_prev = sum(w_saida[i] * todas_h[-1][i] for i in range(n_ocultos)) + b_saida
    return y_prev, todas_h, todos_z

# ============================================================
# BACKWARD PASS
# ============================================================
def backward(X_norm, y_real, todas_h, todos_z):
    dL_dW = [[[0.0]*n_ocultos for _ in range(len(camadas_W[l]))]
              for l in range(n_camadas)]
    dL_db    = [[0.0]*n_ocultos for _ in range(n_camadas)]
    dL_dws   = [0.0] * n_ocultos
    dL_dbs   = 0.0

    y_prev = sum(w_saida[i] * todas_h[-1][i] for i in range(n_ocultos)) + b_saida
    e = y_prev - y_real

    for i in range(n_ocultos):
        dL_dws[i] = 2 * e * todas_h[-1][i]
    dL_dbs = 2 * e

    delta = [2 * e * w_saida[i] for i in range(n_ocultos)]

    for l in range(n_camadas - 1, -1, -1):
        entrada = X_norm if l == 0 else todas_h[l - 1]
        n_entradas = len(entrada)
        novo_delta = [0.0] * n_entradas

        for i in range(n_ocultos):
            d    = d_leaky_relu(todos_z[l][i])
            grad = delta[i] * d
            for j in range(n_entradas):
                dL_dW[l][j][i] += grad * entrada[j]
            dL_db[l][i] += grad
            for j in range(n_entradas):
                novo_delta[j] += grad * camadas_W[l][j][i]

        delta = novo_delta

    return dL_dW, dL_db, dL_dws, dL_dbs

# ============================================================
# TREINAMENTO
# ============================================================
print("\n" + "=" * 60)
print("TREINANDO")
print("=" * 60)
print(f"{'Época':>6}  {'MSE':>10}  {'neurônios vivos':>20}")
print("-" * 45)

for epoca in range(epocas):
    grad_W  = [[[0.0]*n_ocultos for _ in range(len(camadas_W[l]))]
                for l in range(n_camadas)]
    grad_b  = [[0.0]*n_ocultos for _ in range(n_camadas)]
    grad_ws = [0.0] * n_ocultos
    grad_bs = 0.0

    previsoes = []
    for X, y in dataset:
        X_norm = normalizar(X)
        y_prev, todas_h, todos_z = forward(X_norm)
        previsoes.append(y_prev)

        dW, db, dws, dbs = backward(X_norm, y, todas_h, todos_z)

        for l in range(n_camadas):
            for j in range(len(camadas_W[l])):
                for i in range(n_ocultos):
                    grad_W[l][j][i] += dW[l][j][i] / n
        for l in range(n_camadas):
            for i in range(n_ocultos):
                grad_b[l][i] += db[l][i] / n
        for i in range(n_ocultos):
            grad_ws[i] += dws[i] / n
        grad_bs += dbs / n

    # atualização com gradient clipping
    for l in range(n_camadas):
        for j in range(len(camadas_W[l])):
            for i in range(n_ocultos):
                g = clip(grad_W[l][j][i], max_grad)
                camadas_W[l][j][i] -= lr * g
        for i in range(n_ocultos):
            g = clip(grad_b[l][i], max_grad)
            camadas_b[l][i] -= lr * g
    for i in range(n_ocultos):
        w_saida[i] -= lr * clip(grad_ws[i], max_grad)
    b_saida -= lr * clip(grad_bs, max_grad)

    L = mse(previsoes, [y for _, y in dataset])

    if epoca % 300 == 0 or epoca == epocas - 1:
        vivos = sum(
            1 for X, _ in dataset
            for z_camada in forward(normalizar(X))[2]
            for z in z_camada if z > 0
        )
        total_possiveis = n * n_camadas * n_ocultos
        print(f"{epoca:>6}  {L:>10.4f}  {vivos:>8}/{total_possiveis} ({vivos/total_possiveis*100:.0f}%)")

# ============================================================
# INFERÊNCIAS
# ============================================================
print("\n" + "=" * 60)
print("INFERÊNCIAS")
print("=" * 60)
for X, desc in [([1.65,22],"jovem, baixa"), ([1.75,35],"média, média"),
                ([1.90,50],"alta, mais velho"), ([1.60,60],"baixa, mais velho")]:
    y_prev, todas_h, _ = forward(normalizar(X))
    print(f"  altura={X[0]}m  idade={X[1]}a  ({desc:20})  →  {y_prev:.1f} kg")