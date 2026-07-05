"""
TRANSFORMER SIMPLIFICADO vs REDE DENSA — comparativo
======================================================
Objetivo: mostrar como residual connections e layer normalization
resolvem o vanishing gradient que apareceu com 100 camadas densas.

Não é um transformer completo (sem atenção ainda) — é a estrutura
de blocos residuais que é a base do transformer.

Componentes implementados:
    1. Layer Normalization  → normaliza ativações entre blocos
    2. Residual connections → gradiente tem caminho direto
    3. Bloco residual       → [LayerNorm → Dense → Relu] + entrada

Comparativo:
    rede densa 10 camadas   → sem residual, sem layer norm
    rede residual 10 blocos → com residual e layer norm
"""

import random
import math
random.seed(42)

# ============================================================
# CONFIGURAÇÃO
# ============================================================
n_blocos  = 100    # número de blocos (camadas)
n_ocultos = 8     # neurônios por bloco (maior para ver o efeito)
lr        = 0.001
max_grad  = 1.0
epocas    = 3000

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

def relu(x):
    return max(0.0, x)

def drelu(z):
    return 1.0 if z > 0 else 0.0

def mse(previsoes, reais):
    return sum((p - r) ** 2 for p, r in zip(previsoes, reais)) / len(reais)

def clip(v, limite):
    return max(-limite, min(limite, v))

def init_He(n_ent, n_sai):
    escala = math.sqrt(2.0 / n_ent)
    return [[random.gauss(0, escala) for _ in range(n_sai)]
            for _ in range(n_ent)]

# ============================================================
# LAYER NORMALIZATION
# ============================================================
def layer_norm(x, eps=1e-8):
    """
    Normaliza um vetor x para ter média 0 e desvio 1.
    É o z-score que você já conhece, aplicado nas ativações.

    Por que aqui? Para manter os valores em escala controlada
    entre blocos — evita que os valores explodvam ou desapareçam.
    """
    media  = sum(x) / len(x)
    var    = sum((xi - media) ** 2 for xi in x) / len(x)
    desvio = math.sqrt(var + eps)
    return [(xi - media) / desvio for xi in x]

def d_layer_norm(x, grad_out, eps=1e-8):
    """
    Derivada do layer norm — necessária para o backward pass.
    Propaga o gradiente através da normalização.
    """
    n   = len(x)
    media  = sum(x) / n
    var    = sum((xi - media) ** 2 for xi in x) / n
    desvio = math.sqrt(var + eps)
    x_norm = [(xi - media) / desvio for xi in x]

    d_var   = sum(grad_out[i] * x_norm[i] for i in range(n)) * (-0.5) / (var + eps)
    d_media = sum(-grad_out[i] / desvio for i in range(n)) + d_var * sum(-2*(xi-media) for xi in x) / n

    return [grad_out[i]/desvio + d_var*2*(x[i]-media)/n + d_media/n for i in range(n)]


# ============================================================
# PARÂMETROS
# ============================================================
# --- REDE DENSA (sem residual) ---
densa_W = []
densa_b = []
# primeira camada: n_features → n_ocultos
densa_W.append(init_He(n_features, n_ocultos))
densa_b.append([0.0] * n_ocultos)
# demais camadas: n_ocultos → n_ocultos
for _ in range(n_blocos - 1):
    densa_W.append(init_He(n_ocultos, n_ocultos))
    densa_b.append([0.0] * n_ocultos)
# saída
densa_ws = [random.gauss(0, math.sqrt(2.0/n_ocultos)) for _ in range(n_ocultos)]
densa_bs = 0.0

# --- REDE RESIDUAL (com residual connections + layer norm) ---
# projeção de entrada: n_features → n_ocultos
proj_W = init_He(n_features, n_ocultos)
proj_b = [0.0] * n_ocultos

# cada bloco tem seus próprios pesos
res_W = [init_He(n_ocultos, n_ocultos) for _ in range(n_blocos)]
res_b = [[0.0] * n_ocultos for _ in range(n_blocos)]

# saída
res_ws = [random.gauss(0, math.sqrt(2.0/n_ocultos)) for _ in range(n_ocultos)]
res_bs = 0.0


# ============================================================
# FORWARD PASS — REDE DENSA
# ============================================================
def forward_densa(X_norm):
    todas_h, todos_z = [], []
    entrada = X_norm

    for l in range(n_blocos):
        n_ent = len(entrada)
        z = [sum(densa_W[l][j][i]*entrada[j] for j in range(n_ent)) + densa_b[l][i]
             for i in range(n_ocultos)]
        h = [relu(zi) for zi in z]
        todos_z.append(z)
        todas_h.append(h)
        entrada = h

    y = sum(densa_ws[i]*todas_h[-1][i] for i in range(n_ocultos)) + densa_bs
    return y, todas_h, todos_z


# ============================================================
# FORWARD PASS — REDE RESIDUAL
# ============================================================
def forward_residual(X_norm):
    # projeção da entrada para dimensão n_ocultos
    h = [sum(proj_W[j][i]*X_norm[j] for j in range(n_features)) + proj_b[i]
         for i in range(n_ocultos)]

    todas_h    = []   # saída de cada bloco
    todos_z    = []   # pré-ativação de cada bloco
    todos_xnorm = []  # entrada normalizada de cada bloco
    todas_res  = []   # entrada residual de cada bloco (antes do layer norm)

    for l in range(n_blocos):
        residual = h[:]   # guarda entrada para a conexão residual

        # layer normalization
        h_norm = layer_norm(h)
        todos_xnorm.append(h_norm)
        todas_res.append(residual)

        # transformação densa
        z = [sum(res_W[l][j][i]*h_norm[j] for j in range(n_ocultos)) + res_b[l][i]
             for i in range(n_ocultos)]
        todos_z.append(z)

        # relu
        h_new = [relu(zi) for zi in z]
        todas_h.append(h_new)

        # conexão residual: soma a entrada original com a saída transformada
        h = [h_new[i] + residual[i] for i in range(n_ocultos)]

    y = sum(res_ws[i]*h[i] for i in range(n_ocultos)) + res_bs
    return y, h, todas_h, todos_z, todos_xnorm, todas_res


# ============================================================
# BACKWARD — REDE DENSA
# ============================================================
def backward_densa(X_norm, y_real, todas_h, todos_z):
    dW = [[[0.0]*n_ocultos for _ in range(len(densa_W[l]))] for l in range(n_blocos)]
    db = [[0.0]*n_ocultos for _ in range(n_blocos)]
    dws, dbs = [0.0]*n_ocultos, 0.0

    y_prev = sum(densa_ws[i]*todas_h[-1][i] for i in range(n_ocultos)) + densa_bs
    e = y_prev - y_real

    for i in range(n_ocultos):
        dws[i] = 2*e*todas_h[-1][i]
    dbs = 2*e

    delta = [2*e*densa_ws[i] for i in range(n_ocultos)]

    for l in range(n_blocos-1, -1, -1):
        entrada = X_norm if l == 0 else todas_h[l-1]
        n_ent   = len(entrada)
        novo_delta = [0.0]*n_ent
        for i in range(n_ocultos):
            g = delta[i] * drelu(todos_z[l][i])
            for j in range(n_ent):
                dW[l][j][i] += g*entrada[j]
            db[l][i] += g
            for j in range(n_ent):
                novo_delta[j] += g*densa_W[l][j][i]
        delta = novo_delta

    return dW, db, dws, dbs


# ============================================================
# BACKWARD — REDE RESIDUAL
# ============================================================
def backward_residual(X_norm, y_real, h_final, todas_h, todos_z, todos_xnorm, todas_res):
    dW  = [[[0.0]*n_ocultos for _ in range(n_ocultos)] for _ in range(n_blocos)]
    db  = [[0.0]*n_ocultos for _ in range(n_blocos)]
    dpW = [[0.0]*n_ocultos for _ in range(n_features)]
    dpb = [0.0]*n_ocultos
    dws, dbs = [0.0]*n_ocultos, 0.0

    y_prev = sum(res_ws[i]*h_final[i] for i in range(n_ocultos)) + res_bs
    e = y_prev - y_real

    for i in range(n_ocultos):
        dws[i] = 2*e*h_final[i]
    dbs = 2*e

    # gradiente entrando no último bloco vindo da saída
    delta = [2*e*res_ws[i] for i in range(n_ocultos)]

    for l in range(n_blocos-1, -1, -1):
        # conexão residual: gradiente se divide em dois caminhos
        # delta vem da saída do bloco (h_new + residual)
        # parte vai pela transformação, parte vai direto (residual)
        delta_residual    = delta[:]   # caminho direto ← isso é a mágica!
        delta_transformacao = delta[:]

        # backward pelo relu
        grad_relu = [delta_transformacao[i]*drelu(todos_z[l][i]) for i in range(n_ocultos)]

        # backward pela camada densa
        for i in range(n_ocultos):
            for j in range(n_ocultos):
                dW[l][j][i] += grad_relu[i]*todos_xnorm[l][j]
            db[l][i] += grad_relu[i]

        # gradiente antes do layer norm
        grad_pre_norm = [sum(grad_relu[i]*res_W[l][j][i] for i in range(n_ocultos))
                         for j in range(n_ocultos)]

        # backward pelo layer norm
        grad_norm = d_layer_norm(todas_res[l], grad_pre_norm)

        # soma os dois caminhos: transformação + residual direto
        delta = [grad_norm[i] + delta_residual[i] for i in range(n_ocultos)]

    # backward pela projeção de entrada
    for i in range(n_ocultos):
        for j in range(n_features):
            dpW[j][i] += delta[i]*X_norm[j]
        dpb[i] += delta[i]

    return dW, db, dws, dbs, dpW, dpb


# ============================================================
# TREINAMENTO
# ============================================================
def treinar(tipo, epocas):
    print(f"\n{'='*60}")
    print(f"TREINANDO — {tipo}")
    print(f"{'='*60}")
    print(f"{'Época':>6}  {'MSE':>12}")
    print("-" * 25)

    global densa_W, densa_b, densa_ws, densa_bs
    global proj_W, proj_b, res_W, res_b, res_ws, res_bs

    for epoca in range(epocas):
        previsoes = []

        if tipo == "densa":
            gW  = [[[0.0]*n_ocultos for _ in range(len(densa_W[l]))] for l in range(n_blocos)]
            gb  = [[0.0]*n_ocultos for _ in range(n_blocos)]
            gws = [0.0]*n_ocultos
            gbs = 0.0

            for X, y in dataset:
                Xn = normalizar(X)
                yp, th, tz = forward_densa(Xn)
                previsoes.append(yp)
                dW, db, dw, dbias = backward_densa(Xn, y, th, tz)
                for l in range(n_blocos):
                    for j in range(len(densa_W[l])):
                        for i in range(n_ocultos):
                            gW[l][j][i] += dW[l][j][i]/n
                for l in range(n_blocos):
                    for i in range(n_ocultos):
                        gb[l][i] += db[l][i]/n
                for i in range(n_ocultos): gws[i] += dw[i]/n
                gbs += dbias/n

            for l in range(n_blocos):
                for j in range(len(densa_W[l])):
                    for i in range(n_ocultos):
                        densa_W[l][j][i] -= lr*clip(gW[l][j][i], max_grad)
                for i in range(n_ocultos):
                    densa_b[l][i] -= lr*clip(gb[l][i], max_grad)
            for i in range(n_ocultos): densa_ws[i] -= lr*clip(gws[i], max_grad)
            densa_bs -= lr*clip(gbs, max_grad)

        else:  # residual
            gW  = [[[0.0]*n_ocultos for _ in range(n_ocultos)] for _ in range(n_blocos)]
            gb  = [[0.0]*n_ocultos for _ in range(n_blocos)]
            gpW = [[0.0]*n_ocultos for _ in range(n_features)]
            gpb = [0.0]*n_ocultos
            gws = [0.0]*n_ocultos
            gbs = 0.0

            for X, y in dataset:
                Xn = normalizar(X)
                yp, hf, th, tz, txn, tr = forward_residual(Xn)
                previsoes.append(yp)
                dW, db, dw, dbias, dpW, dpb = backward_residual(Xn, y, hf, th, tz, txn, tr)
                for l in range(n_blocos):
                    for j in range(n_ocultos):
                        for i in range(n_ocultos):
                            gW[l][j][i] += dW[l][j][i]/n
                for l in range(n_blocos):
                    for i in range(n_ocultos):
                        gb[l][i] += db[l][i]/n
                for j in range(n_features):
                    for i in range(n_ocultos):
                        gpW[j][i] += dpW[j][i]/n
                for i in range(n_ocultos): gpb[i] += dpb[i]/n
                for i in range(n_ocultos): gws[i] += dw[i]/n
                gbs += dbias/n

            for l in range(n_blocos):
                for j in range(n_ocultos):
                    for i in range(n_ocultos):
                        res_W[l][j][i] -= lr*clip(gW[l][j][i], max_grad)
                for i in range(n_ocultos):
                    res_b[l][i] -= lr*clip(gb[l][i], max_grad)
            for j in range(n_features):
                for i in range(n_ocultos):
                    proj_W[j][i] -= lr*clip(gpW[j][i], max_grad)
            for i in range(n_ocultos): proj_b[i] -= lr*clip(gpb[i], max_grad)
            for i in range(n_ocultos): res_ws[i] -= lr*clip(gws[i], max_grad)
            res_bs -= lr*clip(gbs, max_grad)

        L = mse(previsoes, [y for _, y in dataset])
        if epoca % 300 == 0 or epoca == epocas-1:
            print(f"{epoca:>6}  {L:>12.4f}")

    return L


mse_densa   = treinar("densa",    epocas)
mse_residual = treinar("residual", epocas)

# ============================================================
# COMPARATIVO FINAL
# ============================================================
print(f"\n{'='*60}")
print("COMPARATIVO FINAL")
print(f"{'='*60}")
print(f"  {'':35}  {'densa':>10}  {'residual':>10}")
print(f"  {'camadas':35}  {n_blocos:>10}  {n_blocos:>10}")
print(f"  {'residual connections':35}  {'não':>10}  {'sim':>10}")
print(f"  {'layer normalization':35}  {'não':>10}  {'sim':>10}")
print(f"  {'MSE final':35}  {mse_densa:>10.4f}  {mse_residual:>10.4f}")
print(f"  {'vanishing gradient':35}  {'sim':>10}  {'não':>10}")

print(f"\n{'='*60}")
print("INFERÊNCIAS")
print(f"{'='*60}")
testes = [([1.65,22],"jovem, baixa"), ([1.75,35],"média, média"),
          ([1.90,50],"alta, mais velho"), ([1.60,60],"baixa, mais velho")]

print("\n  REDE DENSA:")
for X, desc in testes:
    yp, _, _ = forward_densa(normalizar(X))
    print(f"    {desc:20}  →  {yp:.1f} kg")

print("\n  REDE RESIDUAL:")
for X, desc in testes:
    yp, *_ = forward_residual(normalizar(X))
    print(f"    {desc:20}  →  {yp:.1f} kg")