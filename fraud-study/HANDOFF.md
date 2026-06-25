# HANDOFF — Transformer Fraud Detection
## Spec-Driven Development — Claude Code

---

## Contexto do projeto

Projeto didático para entender o fluxo completo de um transformer aplicado a detecção de fraude bancária. O objetivo é fixar conceitos estudados — não resolver fraude em produção.

**Quem vai usar:** estudante de ML com conhecimento conceitual sólido de transformers, embeddings, atenção, Q/K/V, softmax, AUROC, AUPRC. Não precisa reexplicar esses conceitos no código — só referenciar.

**Requisito principal:** cada script roda standalone (`python src/01_tokenizer.py`) e imprime output explicativo no terminal. O projeto inteiro roda em CPU, sem GPU.

---

## Stack

```
python 3.9+
torch
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
```

---

## Estrutura de pastas

```
transformer-fraud/
├── README.md
├── requirements.txt
├── HANDOFF.md                  ← este arquivo
├── data/
│   ├── generate_data.py        ← gera transactions.csv
│   └── transactions.csv        ← gerado (não commitar se grande)
├── src/
│   ├── 00_overview.py
│   ├── 01_tokenizer.py
│   ├── 02_embeddings.py
│   ├── 03_local_fusion.py
│   ├── 04_positional.py
│   ├── 05_attention.py
│   ├── 06_multihead.py
│   ├── 07_transformer.py
│   ├── 08_classifier.py
│   └── 09_train_and_eval.py
└── outputs/
    └── .gitkeep
```

---

## Parâmetros globais (config.py ou constantes em cada arquivo)

```python
# Dados
N_CLIENTES   = 200
N_FRAUD      = 60     # 30% de fraude
MAX_EVENTOS  = 12
MIN_EVENTOS  = 5

# Modelo
SEQ_LEN      = 10    # padding/truncate
D_MODEL      = 32    # dimensão dos embeddings
N_HEADS      = 4     # cabeças de atenção
N_LAYERS     = 2     # camadas do transformer
D_FF         = 64    # feed-forward
DROPOUT      = 0.1

# Treino
EPOCHS       = 30
BATCH_SIZE   = 32
LR           = 0.001

# Vocabulários (fixos — não aprendidos)
TIPOS   = {"login":0, "compra":1, "compra_negada":2,
           "troca_senha":3, "consulta_saldo":4,
           "transferencia":5, "saque":6}
PAISES  = {"BR":0, "US":1, "AR":2, "JP":3, "DE":4, "MX":5}
DEVICES = {"celular_habitual":0, "celular_novo":1,
           "desktop_habitual":2, "dispositivo_desconhecido":3}
```

---

## SPEC — data/generate_data.py

### Propósito
Gerar um event stream sintético de transações bancárias com padrões realistas de fraude.

### Output
`data/transactions.csv` com colunas:
```
cliente_id    int   identificador do cliente (0 a N_CLIENTES-1)
evento_num    int   posição na sequência (0, 1, 2...)
tipo          str   tipo do evento
pais          str   país da transação
device        str   dispositivo usado
valor         float valor em R$ (0.0 se não aplicável)
hora          int   hora do dia (0-23)
fase_critica  int   1 se evento suspeito final, 0 caso contrário
fraude        int   label: 1 = fraude, 0 = normal
```

### Comportamento esperado
- Clientes normais: eventos diurnos, país BR, celular habitual, valores baixos
- Clientes com fraude: fase inicial normal, fase final com troca_senha + pais estrangeiro + dispositivo_desconhecido + hora madrugada
- Probabilidades diferentes por perfil para cada campo

### Output no terminal
```
✅ Arquivo salvo: data/transactions.csv
Total de eventos:    1.847
Total de clientes:   200
Clientes com fraude: 60 (30%)
Eventos por cliente: 9.2 em média
── Distribuição de tipos ──
...
── Primeiras linhas ──
...
```

---

## SPEC — src/00_overview.py

### Propósito
Mostrar o fluxo completo do transformer em pseudocódigo comentado. Não implementa nada — serve como mapa do projeto.

### Output no terminal
Diagrama ASCII do fluxo completo:
```
DADO BRUTO
  cliente_id=5, evento_num=6, tipo="troca_senha",
  pais="JP", device="dispositivo_desconhecido",
  valor=0.0, hora=3
       ↓
[BLOCO 1] FIELD ENCODERS (src/02_embeddings.py)
  tipo  → nn.Embedding(7, 32)  → vetor [32]
  pais  → nn.Embedding(6, 32)  → vetor [32]
  device→ nn.Embedding(4, 32)  → vetor [32]
  valor → nn.Linear(1, 32)     → vetor [32]
  hora  → nn.Linear(1, 32)     → vetor [32]
       ↓
[BLOCO 2] LOCAL FUSION (src/03_local_fusion.py)
  concat([tipo, pais, device, valor, hora]) → [160]
  nn.Linear(160, 32) → token_embedding [32]
       ↓
[BLOCO 3] POSITIONAL ENCODING (src/04_positional.py)
  token_embedding + pos_encoding(6) → [32]
  (evento sabe que é o 6º da sequência)
       ↓
[BLOCO 4] MULTI-HEAD ATTENTION (src/06_multihead.py)
  4 cabeças × atenção independente
  cada cabeça aprende um tipo de relação
  concat + projeção Wo → [32]
       ↓
[BLOCO 5] FEED-FORWARD + NORM
  LayerNorm + Linear(32→64→32) + ReLU
       ↓
[BLOCO 6] CLASSIFIER (src/08_classifier.py)
  output[-1] → Linear(32, 1) → sigmoid → score
       ↓
SCORE = 0.94 → FRAUDE DETECTADA
```

---

## SPEC — src/01_tokenizer.py

### Propósito
Converter o CSV bruto em arrays numpy prontos para o modelo. É o pré-processamento — ainda não é o embedding.

### Input
`data/transactions.csv`

### Output
```python
{
  "X": np.array shape [N_CLIENTES, SEQ_LEN, N_CAMPOS],  # features por evento
  "y": np.array shape [N_CLIENTES],                      # labels (0 ou 1)
  "masks": np.array shape [N_CLIENTES, SEQ_LEN],         # 1=evento real, 0=padding
}
```

### Transformações por campo
```
tipo    → TIPOS[tipo]          int   (label encoding)
pais    → PAISES[pais]         int   (label encoding)
device  → DEVICES[device]      int   (label encoding)
valor   → valor / 10000.0      float (normalizado 0-1, clip em 1.0)
hora    → hora / 23.0          float (normalizado 0-1)
```

### Regras de sequência
- Cada cliente tem SEQ_LEN=10 eventos
- Se tiver menos: padding com zeros no início (left-padding)
- Se tiver mais: usar os últimos SEQ_LEN eventos (mais recentes)
- mask=0 indica evento de padding

### Output no terminal
```
✅ Tokenização concluída
Shape X:     (200, 10, 5)
Shape y:     (200,)
Shape masks: (200, 10)
Fraude:      60 (30.0%)

── Exemplo: cliente 5 (fraude=1) ──
evento  tipo  pais  device  valor   hora
0       0     0     0       0.000   0.00   ← padding
1       0     0     0       0.000   0.00   ← padding
2       0     0     0       0.000   0.00   ← padding
3       0     0     0       0.000   0.00   ← padding
4       1     0     0       0.012   0.61   ← compra BR
5       0     0     0       0.000   0.52   ← login BR
6       3     3     3       0.000   0.13   ← troca_senha JP 3h  ⚠
7       2     1     3       0.000   0.96   ← compra_negada US   ⚠
8       1     3     3       0.420   0.17   ← compra JP R$4.200  🚨
9       5     3     3       0.380   0.22   ← transferencia JP   🚨
```

---

## SPEC — src/02_embeddings.py

### Propósito
Transformar os índices numéricos em vetores densos de D_MODEL dimensões. Demonstrar visualmente a diferença entre um token normal e um suspeito no espaço vetorial.

### Implementação
```python
class FieldEncoders(nn.Module):
    # Um nn.Embedding por campo categórico
    # Um nn.Linear por campo numérico
    # Cada um projeta para D_MODEL dims
```

### Output no terminal
```
── Embedding de evento NORMAL (login BR celular 14h) ──
tipo  [32 dims]: [ 0.12, -0.03,  0.44, ... ]
pais  [32 dims]: [ 0.71,  0.22, -0.55, ... ]
device[32 dims]: [ 0.82, -0.11,  0.33, ... ]
valor [32 dims]: [ 0.00,  0.00,  0.00, ... ]  ← zero (sem valor)
hora  [32 dims]: [ 0.55, -0.21,  0.44, ... ]

── Embedding de evento SUSPEITO (troca_senha JP dispositivo_desc 3h) ──
tipo  [32 dims]: [ 0.91, -0.88,  1.05, ... ]  ← valores diferentes
pais  [32 dims]: [ 0.72,  0.55, -0.33, ... ]
device[32 dims]: [ 0.95, -0.71,  0.88, ... ]  ← muito diferente
valor [32 dims]: [ 0.00,  0.00,  0.00, ... ]
hora  [32 dims]: [ 0.88, -0.12,  0.77, ... ]  ← madrugada

Distância L2 entre os dois eventos: 4.82
(tokens diferentes ficam distantes no espaço vetorial)
```

---

## SPEC — src/03_local_fusion.py

### Propósito
Combinar os embeddings dos 5 campos de um evento em um único vetor de D_MODEL. Demonstrar que a combinação captura mais informação do que qualquer campo isolado.

### Implementação
```python
class LocalFusion(nn.Module):
    # concat([emb_tipo, emb_pais, emb_device, emb_valor, emb_hora])
    # → shape [D_MODEL * 5] = [160]
    # nn.Linear(160, D_MODEL) → shape [32]
    # ReLU + LayerNorm
```

### Output no terminal
```
── Campo isolado: tipo="troca_senha" ──
vetor [32]: [ 0.91, -0.43,  0.88, ... ]
norma: 3.21

── Campo isolado: pais="JP" ──
vetor [32]: [ 0.72,  0.55, -0.33, ... ]
norma: 2.87

── Campo isolado: device="dispositivo_desconhecido" ──
vetor [32]: [ 0.95, -0.71,  0.88, ... ]
norma: 3.45

── Após local fusion (troca_senha + JP + dispositivo_desc + 3h) ──
vetor [32]: [ 1.42,  0.98, -1.21, ... ]
norma: 5.83  ← muito maior — combinação amplifica o sinal

Conclusão: a norma do vetor fundido (5.83) é maior
que qualquer campo isolado. A combinação de campos
suspeitos cria um sinal mais intenso.
```

---

## SPEC — src/04_positional.py

### Propósito
Adicionar informação de posição temporal ao embedding de cada evento. Demonstrar que o transformer precisa saber a ordem dos eventos.

### Implementação
Positional encoding sinusoidal (Vaswani et al., 2017):
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Output no terminal
```
── Positional encoding para posição 0 (primeiro evento) ──
[32 dims]: [ 0.00,  1.00,  0.00,  1.00, ... ]

── Positional encoding para posição 6 (sétimo evento) ──
[32 dims]: [ 0.28,  0.96,  0.06,  1.00, ... ]

── Embedding ANTES do positional encoding (evento 6) ──
[32 dims]: [ 0.91, -0.43,  0.88, ... ]

── Embedding DEPOIS do positional encoding (evento 6) ──
[32 dims]: [ 1.19,  0.53,  0.94, ... ]  ← posição somada

Sem positional encoding, o transformer trataria
"troca_senha no evento 2" igual a "troca_senha no evento 9".
A posição importa — um evento suspeito no final da sequência
é mais relevante do que no início.
```

---

## SPEC — src/05_attention.py

### Propósito
Implementar single-head attention do zero com números pequenos. Mostrar a matriz de atenção e os pesos que emergem para um cliente de fraude.

### Implementação
```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights
```

### Output no terminal
```
── Sequência do cliente 5 (fraude) ──
pos  evento
0    padding
1    padding
2    padding
3    padding
4    compra BR
5    login BR
6    troca_senha JP  ⚠
7    compra_negada   ⚠
8    compra JP       🚨
9    transferencia   🚨

── Matriz de atenção (evento 9 olhando para todos) ──
       pos0  pos1  pos2  pos3  pos4  pos5  pos6  pos7  pos8  pos9
peso:  0.00  0.00  0.00  0.00  0.02  0.03  0.31  0.28  0.22  0.14

Interpretação:
  pos6 (troca_senha JP): 31% da atenção  ← evento mais relevante
  pos7 (compra_negada):  28% da atenção
  pos8 (compra JP):      22% da atenção
  pos4-5 (normais):       5% da atenção
  padding:                0% da atenção
```

---

## SPEC — src/06_multihead.py

### Propósito
Estender para N_HEADS=4 cabeças em paralelo. Mostrar que cada cabeça aprende pesos de atenção diferentes para a mesma sequência.

### Implementação
```python
class MultiHeadAttention(nn.Module):
    # Para cada cabeça i:
    #   Qi = Linear(D_MODEL, D_HEAD)(X)
    #   Ki = Linear(D_MODEL, D_HEAD)(X)
    #   Vi = Linear(D_MODEL, D_HEAD)(X)
    #   head_i = attention(Qi, Ki, Vi)
    # output = Linear(D_MODEL, D_MODEL)(concat(head_1...head_N))
```

### Output no terminal
```
── 4 cabeças de atenção para o evento 9 (transferencia JP) ──

Cabeça 1 — padrão temporal:
  maior atenção: pos6=0.38, pos7=0.31

Cabeça 2 — padrão geográfico:
  maior atenção: pos6=0.44, pos8=0.29

Cabeça 3 — padrão de valor:
  maior atenção: pos8=0.41, pos9=0.33

Cabeça 4 — padrão de dispositivo:
  maior atenção: pos6=0.35, pos7=0.30

Cada cabeça encontrou padrões diferentes
na mesma sequência — isso é multi-head attention.
```

---

## SPEC — src/07_transformer.py

### Propósito
Montar o temporal backbone completo: embedding + positional + N camadas de (multi-head attention + feed-forward + layer norm).

### Implementação
```python
class FraudTransformer(nn.Module):
    def __init__(self):
        self.field_encoders  = FieldEncoders(...)
        self.local_fusion    = LocalFusion(...)
        self.pos_encoding    = PositionalEncoding(...)
        self.layers          = nn.ModuleList([
            TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT)
            for _ in range(N_LAYERS)
        ])
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, n_campos]
        # 1. field encoders por campo
        # 2. local fusion → [batch, seq_len, D_MODEL]
        # 3. + positional encoding
        # 4. N layers de atenção + FF
        # 5. return output: [batch, seq_len, D_MODEL]
```

### Output no terminal
```
FraudTransformer(
  (field_encoders): FieldEncoders(tipo=Emb(7,32), pais=Emb(6,32), ...)
  (local_fusion):   LocalFusion(Linear(160→32))
  (pos_encoding):   PositionalEncoding(d=32, max_len=10)
  (layers):         2x TransformerEncoderLayer(d=32, heads=4, ff=64)
)

Total de parâmetros: 24.897  ← pequeno, roda em CPU

── Forward pass (1 cliente, 10 eventos) ──
Input shape:   [1, 10, 5]
After encoders:[1, 10, 32]
After fusion:  [1, 10, 32]
After pos enc: [1, 10, 32]
After layer 1: [1, 10, 32]
After layer 2: [1, 10, 32]  ← output final
```

---

## SPEC — src/08_classifier.py

### Propósito
Converter o output do transformer em um score de fraude via sigmoide. Conectar com o conceito de z = w·x + b que foi estudado.

### Implementação
```python
class FraudClassifier(nn.Module):
    def __init__(self):
        self.transformer = FraudTransformer()
        self.classifier  = nn.Sequential(
            nn.Linear(D_MODEL, 1),  # z = w·x + b
            nn.Sigmoid()            # score = 1/(1+e^-z)
        )

    def forward(self, x, mask=None):
        output = self.transformer(x, mask)
        last   = output[:, -1, :]   # último token não-padding
        score  = self.classifier(last)
        return score
```

### Output no terminal
```
── Score para cliente 5 (fraude real = 1) ──
z (antes da sigmoide): 2.41
score (após sigmoide): 0.918  → predito como FRAUDE ✅

── Score para cliente 150 (fraude real = 0) ──
z (antes da sigmoide): -1.83
score (após sigmoide): 0.138  → predito como NORMAL ✅

── Score para cliente 30 (fraude real = 1, difícil) ──
z (antes da sigmoide): 0.21
score (após sigmoide): 0.552  → predito como FRAUDE ✅ (por pouco)
```

---

## SPEC — src/09_train_and_eval.py

### Propósito
Treinar o transformer e comparar com XGBoost clássico. Mostrar onde cada modelo acerta e erra, e por quê.

### Fluxo
```
1. Carregar dados (01_tokenizer)
2. Split treino/teste 80/20 estratificado
3. Treinar FraudClassifier (transformer)
4. Treinar XGBoost com features agregadas (modelo clássico)
5. Avaliar os dois com AUROC, AUPRC, matriz de confusão
6. Mostrar exemplos de onde o transformer acerta e o clássico erra
7. Mostrar os pesos de atenção dos casos mais interessantes
```

### Features do XGBoost (agregações)
```python
# O que o modelo clássico vê — 1 linha por cliente
features_classico = {
    "n_eventos":          len(eventos),
    "n_troca_senha":      count("troca_senha"),
    "n_compra_negada":    count("compra_negada"),
    "n_pais_estrangeiro": count(pais != "BR"),
    "n_device_desc":      count("dispositivo_desconhecido"),
    "valor_medio":        mean(valor),
    "valor_max":          max(valor),
    "hora_media":         mean(hora),
    "n_madrugada":        count(hora < 6),
    "n_fase_critica":     count(fase_critica == 1),
}
```

### Output no terminal
```
══════════════════════════════════════════════════
RESULTADOS FINAIS — Transformer vs XGBoost
══════════════════════════════════════════════════

Modelo               AUROC   AUPRC   Prec    Rec     F1
─────────────────────────────────────────────────────
XGBoost (clássico)   0.XXX   0.XXX   0.XXX   0.XXX   0.XXX
Transformer (LDM)    0.XXX   0.XXX   0.XXX   0.XXX   0.XXX

── Casos onde o Transformer acerta e o XGBoost erra ──
Cliente 12: fraude real=1
  XGBoost score: 0.31  → disse NORMAL  ❌
  Transformer:   0.87  → disse FRAUDE  ✅
  Por quê? A sequência tinha troca_senha no evento 3
  seguida de compra normal — o XGBoost não viu a ordem.
  O transformer prestou 42% de atenção no evento 3.

── Pesos de atenção — cliente 12 ──
evento  tipo            atenção
0       padding         0.00
1       login BR        0.02
2       compra BR       0.04
3       troca_senha JP  0.42  ← atenção alta
4       compra normal   0.08
...
```

---

## SPEC — README.md

### Conteúdo
```markdown
# Transformer Fraud Detection — Didático

Implementação passo a passo de um transformer para detecção de fraude,
com dados sintéticos. Objetivo: entender o fluxo completo, não resolver
fraude em produção.

## Setup

git clone <repo>
cd transformer-fraud
python -m venv venv
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt

## Executar

# 1. Gerar dados
python data/generate_data.py

# 2. Explorar o fluxo (overview)
python src/00_overview.py

# 3. Rodar cada bloco em ordem
python src/01_tokenizer.py
python src/02_embeddings.py
...
python src/09_train_and_eval.py

## Estrutura
[tabela de arquivos e o que cada um faz]

## Conceitos implementados
- Field encoders (nn.Embedding + nn.Linear)
- Local fusion (concatenação + projeção linear)
- Positional encoding (sinusoidal)
- Self-attention (Q, K, V, softmax)
- Multi-head attention (N cabeças paralelas)
- Temporal backbone (N camadas empilhadas)
- Score via sigmoide
- Comparação com modelo clássico (XGBoost)
```

---

## Instruções para o Claude Code

1. Criar a estrutura de pastas exatamente como descrita
2. Implementar cada arquivo seguindo sua SPEC
3. Cada arquivo deve rodar standalone e produzir o output descrito
4. Comentários devem explicar O QUE e POR QUÊ — não apenas O QUE
5. Usar as constantes globais definidas em cada arquivo (ou importar de config.py)
6. O arquivo 09 importa os módulos dos outros arquivos
7. Não usar GPU — só CPU
8. requirements.txt deve incluir versões fixas para reprodutibilidade
