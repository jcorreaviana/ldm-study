"""
PIPELINE COMPLETO — Pré-processamento + Mecanismo de Atenção
=============================================================
Implementação do fluxo completo sem nenhuma biblioteca de ML.

Fluxo:
    dados brutos (valor, hora, país, device)
        ↓
    1. z-score para campos numéricos contínuos
    2. one-hot encoding para campos categóricos
    3. binário para campos sim/não
        ↓
    token [8 dimensões]
        ↓
    4. mecanismo de atenção (Q · Kᵀ / √d → softmax → × V)
        ↓
    representação enriquecida com contexto
        ↓
    5. sigmoid → score de fraude entre 0 e 1

Contexto: detecção de fraude em transações financeiras
"""

import math
import random
random.seed(42)

# ============================================================
# HISTÓRICO DO CLIENTE — usado para calcular média e desvio
# ============================================================
historico_valor = [850, 1200, 450, 2100, 680, 920, 4200, 310, 1500, 760]
historico_hora  = [14.5, 10.2, 18.3, 9.1, 15.8, 11.4, 4.37, 16.2, 13.5, 20.1]

# ============================================================
# ETAPA 1 — PRÉ-PROCESSAMENTO
# ============================================================

def calcular_media(valores):
    return sum(valores) / len(valores)

def calcular_desvio(valores):
    media = calcular_media(valores)
    variancia = sum((x - media) ** 2 for x in valores) / len(valores)
    return math.sqrt(variancia)

def z_score(valor, media, desvio):
    """
    Normaliza um valor numérico contínuo.
    z = (valor - média) / desvio
    resultado: número de desvios padrão acima/abaixo da média
    """
    return (valor - media) / desvio

# vocabulário de países para one-hot
PAISES = ['brasil', 'eua', 'europa', 'asia', 'outro']

def one_hot_pais(pais):
    """
    Converte país em vetor binário.
    ex: 'asia' → [0, 0, 0, 1, 0]
    """
    return [1 if p == pais else 0 for p in PAISES]

def binario_device(device):
    """
    Converte device em 0 ou 1.
    conhecido=1, desconhecido=0
    """
    return 1 if device == 'conhecido' else 0

# calcula estatísticas do histórico
media_valor  = calcular_media(historico_valor)
desvio_valor = calcular_desvio(historico_valor)
media_hora   = calcular_media(historico_hora)
desvio_hora  = calcular_desvio(historico_hora)

print("=" * 60)
print("ESTATÍSTICAS DO HISTÓRICO")
print("=" * 60)
print(f"  valor:  média={media_valor:.0f}  desvio={desvio_valor:.0f}")
print(f"  hora:   média={media_hora:.2f}h  desvio={desvio_hora:.2f}h")

def evento_para_token(evento):
    """
    Converte um evento bruto em vetor de 8 dimensões.

    estrutura do token:
    [z_valor, z_hora, brasil, eua, europa, asia, outro, device]
      [0]      [1]    [2]    [3]   [4]    [5]   [6]    [7]
    """
    z_val  = z_score(evento['valor'], media_valor, desvio_valor)
    z_hor  = z_score(evento['hora'],  media_hora,  desvio_hora)
    oh_pais = one_hot_pais(evento['pais'])
    bin_dev = binario_device(evento['device'])

    return [z_val, z_hor] + oh_pais + [bin_dev]

# ============================================================
# SEQUÊNCIA DE EVENTOS DO CLIENTE
# ============================================================
eventos = [
    {'nome': 'login_normal',                'valor': 0,    'hora': 9.5,  'pais': 'brasil', 'device': 'conhecido'},
    {'nome': 'compra_sao_paulo',            'valor': 850,  'hora': 14.2, 'pais': 'brasil', 'device': 'conhecido'},
    {'nome': 'troca_senha_ip_estrangeiro',  'valor': 0,    'hora': 3.14, 'pais': 'asia',   'device': 'desconhecido'},
    {'nome': 'compra_toquio_4h22',          'valor': 4200, 'hora': 4.37, 'pais': 'asia',   'device': 'conhecido'},
]

print("\n" + "=" * 60)
print("ETAPA 1 — PRÉ-PROCESSAMENTO: eventos → tokens")
print("=" * 60)

tokens = []
for evento in eventos:
    token = evento_para_token(evento)
    tokens.append(token)
    print(f"\n  {evento['nome']}:")
    print(f"    valor={evento['valor']} → z={token[0]:+.2f}  hora={evento['hora']} → z={token[1]:+.2f}")
    print(f"    país={evento['pais']} → {token[2:7]}  device={evento['device']} → {token[7]}")
    print(f"    token: [{', '.join(f'{v:+.2f}' for v in token)}]")

n_tokens = len(tokens)
d_model  = len(tokens[0])   # 8 dimensões
d_k      = 4                 # dimensão de Q, K, V

# ============================================================
# FUNÇÕES DO MECANISMO DE ATENÇÃO
# ============================================================

def multiplicar_matrizes(A, B):
    m, n, p = len(A), len(A[0]), len(B[0])
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpor(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def softmax_linha(scores):
    # subtrai o máximo para estabilidade numérica
    max_s = max(scores)
    exp_scores = [math.exp(s - max_s) for s in scores]
    soma = sum(exp_scores)
    return [e / soma for e in exp_scores]

def softmax_matriz(M):
    return [softmax_linha(linha) for linha in M]

def init_matriz(linhas, colunas):
    escala = math.sqrt(2.0 / linhas)
    return [[random.gauss(0, escala) for _ in range(colunas)]
            for _ in range(linhas)]

# ============================================================
# ETAPA 2 — CRIAR Q, K, V
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 2 — Criando Q, K, V")
print("=" * 60)
print(f"  tokens [{n_tokens} × {d_model}]  ×  Wq/Wk/Wv [{d_model} × {d_k}]  =  Q/K/V [{n_tokens} × {d_k}]")

Wq = init_matriz(d_model, d_k)
Wk = init_matriz(d_model, d_k)
Wv = init_matriz(d_model, d_k)

Q = multiplicar_matrizes(tokens, Wq)
K = multiplicar_matrizes(tokens, Wk)
V = multiplicar_matrizes(tokens, Wv)

# ============================================================
# ETAPA 3 — SCORES: Q · Kᵀ / √d
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 3 — Scores de atenção: Q · Kᵀ / √d")
print("=" * 60)

Kt = transpor(K)
scores_raw = multiplicar_matrizes(Q, Kt)
escala = math.sqrt(d_k)
scores = [[s / escala for s in linha] for linha in scores_raw]

print(f"\n  scores (relevância entre cada par de eventos):")
print(f"  {'':35}", end="")
for e in eventos:
    print(f"  {e['nome'][:10]:10}", end="")
print()
for i, (evento, linha) in enumerate(zip(eventos, scores)):
    print(f"  {evento['nome'][:35]:35}", end="")
    for s in linha:
        print(f"  {s:10.3f}", end="")
    print()

# ============================================================
# ETAPA 4 — SOFTMAX → pesos de atenção
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 4 — Softmax → pesos de atenção")
print("=" * 60)

atencao = softmax_matriz(scores)

print(f"\n  pesos de atenção (cada linha soma 1.0):")
print(f"  {'':35}", end="")
for e in eventos:
    print(f"  {e['nome'][:10]:10}", end="")
print()
for i, (evento, linha) in enumerate(zip(eventos, atencao)):
    print(f"  {evento['nome'][:35]:35}", end="")
    for a in linha:
        print(f"  {a:10.3f}", end="")
    print()

# ============================================================
# ETAPA 5 — SAÍDA: atenção · V
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 5 — Saída enriquecida: atenção · V")
print("=" * 60)

saida = multiplicar_matrizes(atencao, V)

print(f"\n  representação final de cada evento (enriquecida com contexto):")
for i, (evento, s) in enumerate(zip(eventos, saida)):
    print(f"  {evento['nome'][:35]:35}: [{', '.join(f'{v:+.3f}' for v in s)}]")

# ============================================================
# INTERPRETAÇÃO — foco no evento de fraude
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETAÇÃO — atenção da compra em Tóquio")
print("=" * 60)

token_alvo = 3
pesos = atencao[token_alvo]
print(f"\n  Quando o modelo avalia '{eventos[token_alvo]['nome']}':")
print(f"  distribui atenção assim:\n")
for i, (evento, peso) in enumerate(zip(eventos, pesos)):
    barra = "█" * int(peso * 50)
    print(f"  {evento['nome'][:35]:35}  {peso:.3f}  {barra}")

# ============================================================
# RESUMO DO PIPELINE COMPLETO
# ============================================================
print("\n" + "=" * 60)
print("RESUMO DO PIPELINE COMPLETO")
print("=" * 60)
print(f"""
  dados brutos (valor, hora, país, device)
       ↓
  1. z-score         →  valor e hora na mesma escala
  2. one-hot         →  país em vetor binário [5 dimensões]
  3. binário         →  device conhecido=1, desconhecido=0
       ↓
  token [{d_model} dimensões] por evento
       ↓
  4. Q, K, V = token × Wq, Wk, Wv    [{n_tokens} × {d_k}]
  5. scores  = Q · Kᵀ / √{d_k}          [{n_tokens} × {n_tokens}]
  6. atenção = softmax(scores)         [{n_tokens} × {n_tokens}]
  7. saída   = atenção · V            [{n_tokens} × {d_k}]
       ↓
  representação enriquecida com contexto
  (cada evento "sabe" o que aconteceu nos outros)
       ↓
  sigmoid → score de fraude 0-1
""")