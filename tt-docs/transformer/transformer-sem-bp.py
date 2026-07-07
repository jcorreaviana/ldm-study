"""
BLOCO TRANSFORMER COMPLETO — do zero
======================================
Implementação completa de um bloco transformer sem biblioteca de ML.

Arquitetura de cada bloco:
    entrada
        ↓
    [Multi-Head Attention]
        ↓
    (+) residual connection
        ↓
    [Layer Normalization]
        ↓
    [Feed-Forward: d_model → 4×d_model → d_model]
        ↓
    (+) residual connection
        ↓
    [Layer Normalization]
        ↓
    saída (mesma dimensão da entrada)

N blocos empilhados em sequência.
Entrada e saída sempre [n_tokens × d_model].

Contexto: detecção de fraude em sequência de eventos
"""

import math
import random
random.seed(42)

# ============================================================
# CONFIGURAÇÃO
# ============================================================
n_tokens  = 4     # eventos na sequência
d_model   = 8     # dimensão de cada token
d_k       = 4     # dimensão de Q, K, V por cabeça
n_heads   = 2     # número de cabeças de atenção
d_ff      = 32    # dimensão interna do feed-forward (4 × d_model)
n_blocos  = 2     # número de blocos transformer empilhados

# ============================================================
# TOKENS DE ENTRADA (pré-processados)
# ============================================================
# Reusa os tokens do pipeline_completo.py
# [z_valor, z_hora, brasil, eua, europa, asia, outro, device]
tokens = [
    [-1.19, -0.83,  1, 0, 0, 0, 0, 1],   # login_normal
    [-0.41,  0.18,  1, 0, 0, 0, 0, 1],   # compra_sao_paulo
    [-1.19, -2.21,  0, 0, 0, 1, 0, 0],   # troca_senha_ip_estrangeiro
    [ 2.67, -1.95,  0, 0, 0, 1, 0, 1],   # compra_toquio
]

eventos = [
    "login_normal",
    "compra_sao_paulo",
    "troca_senha_ip_estrangeiro",
    "compra_toquio_4h22",
]

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def mat_mul(A, B):
    """Multiplicação de matrizes A[m×n] · B[n×p] = C[m×p]"""
    m, n, p = len(A), len(A[0]), len(B[0])
    C = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
         for i in range(m)]
    return C

def transpor(A):
    """Transposta de A[m×n] → Aᵀ[n×m]"""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def softmax_linha(scores):
    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    soma  = sum(exp_s)
    return [e / soma for e in exp_s]

def softmax_mat(M):
    return [softmax_linha(linha) for linha in M]

def relu(x):
    return max(0.0, x)

def layer_norm(x, eps=1e-8):
    """Z-score aplicado nas ativações de cada token"""
    media  = sum(x) / len(x)
    var    = sum((xi - media) ** 2 for xi in x) / len(x)
    desvio = math.sqrt(var + eps)
    return [(xi - media) / desvio for xi in x]

def init_W(linhas, colunas):
    """Inicialização He"""
    escala = math.sqrt(2.0 / linhas)
    return [[random.gauss(0, escala) for _ in range(colunas)]
            for _ in range(linhas)]

def somar_vetores(a, b):
    """Soma elemento a elemento — usada na residual connection"""
    return [a[i] + b[i] for i in range(len(a))]

def somar_matrizes(A, B):
    """Soma de matrizes elemento a elemento"""
    return [somar_vetores(A[i], B[i]) for i in range(len(A))]

# ============================================================
# COMPONENTE 1 — SINGLE-HEAD ATTENTION
# ============================================================
def single_head_attention(X, Wq, Wk, Wv):
    """
    Uma cabeça de atenção.
    X: tokens [n_tokens × d_model]
    retorna: saída [n_tokens × d_k]
    """
    Q = mat_mul(X, Wq)                    # [n × d_k]
    K = mat_mul(X, Wk)                    # [n × d_k]
    V = mat_mul(X, Wv)                    # [n × d_k]

    scores = mat_mul(Q, transpor(K))      # [n × n]
    escala = math.sqrt(d_k)
    scores = [[s / escala for s in linha] for linha in scores]
    atencao = softmax_mat(scores)         # [n × n]

    return mat_mul(atencao, V)            # [n × d_k]

# ============================================================
# COMPONENTE 2 — MULTI-HEAD ATTENTION
# ============================================================
def multi_head_attention(X, cabecas_W):
    """
    N cabeças de atenção em paralelo.
    Cada cabeça aprende um aspecto diferente.
    Concatena as saídas e projeta de volta para d_model.

    cabecas_W: lista de (Wq, Wk, Wv, Wo) por cabeça
    retorna: saída [n_tokens × d_model]
    """
    saidas_cabecas = []

    for Wq, Wk, Wv, Wo in cabecas_W:
        # cada cabeça calcula sua própria atenção
        saida_cabeca = single_head_attention(X, Wq, Wk, Wv)  # [n × d_k]
        # projeta de volta para d_model
        saida_proj = mat_mul(saida_cabeca, Wo)                # [n × d_model]
        saidas_cabecas.append(saida_proj)

    # soma as contribuições de todas as cabeças
    # (simplificação: na prática concatena e projeta)
    resultado = saidas_cabecas[0]
    for s in saidas_cabecas[1:]:
        resultado = somar_matrizes(resultado, s)

    return resultado   # [n_tokens × d_model]

# ============================================================
# COMPONENTE 3 — FEED-FORWARD
# ============================================================
def feed_forward(X, W1, b1, W2, b2):
    """
    Duas camadas densas com relu no meio.
    Aplicado a cada token independentemente.

    d_model → d_ff → d_model
    (expansão e compressão)

    X: tokens [n_tokens × d_model]
    retorna: saída [n_tokens × d_model]
    """
    saida = []
    for token in X:
        # camada 1: expande de d_model para d_ff
        z1 = [sum(token[j] * W1[j][i] for j in range(d_model)) + b1[i]
              for i in range(d_ff)]
        h1 = [relu(z) for z in z1]   # ← relu aqui, não na saída

        # camada 2: comprime de d_ff para d_model
        z2 = [sum(h1[j] * W2[j][i] for j in range(d_ff)) + b2[i]
              for i in range(d_model)]

        saida.append(z2)   # sem relu na saída

    return saida   # [n_tokens × d_model]

# ============================================================
# BLOCO TRANSFORMER COMPLETO
# ============================================================
def bloco_transformer(X, cabecas_W, W1, b1, W2, b2):
    """
    Um bloco completo do transformer:

    1. Multi-Head Attention
    2. Residual connection + Layer Norm
    3. Feed-Forward
    4. Residual connection + Layer Norm

    X: tokens [n_tokens × d_model]
    retorna: saída [n_tokens × d_model]  ← mesma dimensão!
    """
    # ── SUB-BLOCO 1: atenção ──────────────────────────────
    # multi-head attention
    atencao_saida = multi_head_attention(X, cabecas_W)

    # residual connection: soma entrada com saída da atenção
    residual_1 = somar_matrizes(X, atencao_saida)

    # layer normalization em cada token
    norm_1 = [layer_norm(token) for token in residual_1]

    # ── SUB-BLOCO 2: feed-forward ─────────────────────────
    # feed-forward aplicado token por token
    ff_saida = feed_forward(norm_1, W1, b1, W2, b2)

    # residual connection: soma norm_1 com saída do feed-forward
    residual_2 = somar_matrizes(norm_1, ff_saida)

    # layer normalization final
    norm_2 = [layer_norm(token) for token in residual_2]

    return norm_2   # [n_tokens × d_model]

# ============================================================
# INICIALIZAÇÃO DOS PARÂMETROS
# ============================================================

# parâmetros das cabeças de atenção
# cada cabeça tem: Wq, Wk, Wv [d_model × d_k] e Wo [d_k × d_model]
cabecas_W = [
    (init_W(d_model, d_k),    # Wq
     init_W(d_model, d_k),    # Wk
     init_W(d_model, d_k),    # Wv
     init_W(d_k, d_model))    # Wo — projeta de volta para d_model
    for _ in range(n_heads)
]

# parâmetros do feed-forward por bloco
ff_params = [
    (init_W(d_model, d_ff),   # W1 [d_model × d_ff]
     [0.0] * d_ff,            # b1
     init_W(d_ff, d_model),   # W2 [d_ff × d_model]
     [0.0] * d_model)         # b2
    for _ in range(n_blocos)
]

# ============================================================
# FORWARD PASS — N BLOCOS EMPILHADOS
# ============================================================
print("=" * 60)
print("TRANSFORMER COMPLETO — forward pass")
print("=" * 60)
print(f"  tokens:   {n_tokens} eventos  ×  {d_model} dimensões")
print(f"  cabeças:  {n_heads}")
print(f"  blocos:   {n_blocos}")
print(f"  d_ff:     {d_ff}  ({d_ff//d_model}× d_model)")
print()

X = tokens   # entrada inicial

for bloco_idx in range(n_blocos):
    print(f"── Bloco {bloco_idx + 1} ──────────────────────────────────────")

    W1, b1, W2, b2 = ff_params[bloco_idx]
    X = bloco_transformer(X, cabecas_W, W1, b1, W2, b2)

    print(f"  saída [{n_tokens} × {d_model}]:")
    for i, (evento, token) in enumerate(zip(eventos, X)):
        print(f"    token {i} ({evento[:25]:25}): "
              f"[{', '.join(f'{v:+.2f}' for v in token)}]")
    print()

# ============================================================
# SAÍDA FINAL — sigmoid para score de fraude
# ============================================================
print("=" * 60)
print("SAÍDA FINAL — score de fraude por evento")
print("=" * 60)

# pesos de classificação — projeta d_model → 1 número
W_class = [random.gauss(0, math.sqrt(2.0/d_model)) for _ in range(d_model)]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

print()
for i, (evento, token) in enumerate(zip(eventos, X)):
    z = sum(token[j] * W_class[j] for j in range(d_model))
    score = sigmoid(z)
    risco = "🔴 ALTO RISCO" if score > 0.5 else "🟢 baixo risco"
    print(f"  {evento[:35]:35}  score={score:.3f}  {risco}")

# ============================================================
# RESUMO DA ARQUITETURA
# ============================================================
total_params = (
    n_heads * (3 * d_model * d_k + d_k * d_model) +   # atenção
    n_blocos * (d_model * d_ff + d_ff + d_ff * d_model + d_model) +  # ff
    d_model  # classificação
)

print(f"""
{"=" * 60}
RESUMO DA ARQUITETURA
{"=" * 60}

  entrada:    tokens [{n_tokens} × {d_model}]
      ↓
  bloco 1:
    multi-head attention ({n_heads} cabeças)
    residual + layer norm
    feed-forward ({d_model} → {d_ff} → {d_model})
    residual + layer norm
      ↓
  bloco 2:    (mesmo padrão)
      ↓
  sigmoid →   score de fraude por evento

  total de parâmetros (estimado): {total_params}

  o que cada componente faz:
    atenção      →  quais eventos são relevantes entre si?
    feed-forward →  o que fazer com essa informação combinada?
    residual     →  gradiente tem caminho direto (sem vanishing)
    layer norm   →  mantém ativações em escala controlada
""")