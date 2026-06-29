# Resumo de Estudo — Transformer para Detecção de Fraude Bancária

---

## Pipeline Completo (visão geral)

```
CSV bruto
  ↓ 01_tokenizer.py     → números crus          [200, 10, 5]
  ↓ 02_embeddings.py    → vetores por campo      [200, 10, 5, 32]
  ↓ 03_local_fusion.py  → 1 token por evento     [200, 10, 32]
  ↓ 04_positional.py    → token + posição        [200, 10, 32]
  ↓ 05/06_attention.py  → contexto entre eventos [200, 10, 32]
  ↓ 07_transformer.py   → 2 camadas empilhadas   [200, 10, 32]
  ↓ 08_classifier.py    → 1 score por cliente    [200]  ∈ [0, 1]
```

> **Regra de ouro:** o shape `[200, 10, 32]` se mantém ao longo de quase todo o pipeline. O que muda é o *conteúdo* dos vetores, não a estrutura.

---

## Arquivo por Arquivo

---

### config.py — Configuração central

**O que faz:**
- Fixa seed de aleatoriedade (`SEED = 42`) → resultados reproduzíveis
- Define todas as constantes globais do modelo e do treino
- Define os vocabulários fixos (string → índice inteiro)

**Constantes-chave:**

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `D_MODEL` | 32 | Dimensão dos vetores de embedding |
| `N_HEADS` | 4 | Número de cabeças de atenção |
| `D_FF` | 64 | Dimensão interna do feed-forward (4 × D_MODEL) |
| `N_LAYERS` | 2 | Número de camadas do transformer |
| `SEQ_LEN` | 10 | Eventos por cliente |
| `N_CLIENTES` | 200 | Total de clientes |

**Por que D_MODEL = 32?**
- Campo mais rico: `TIPOS` tem 7 categorias
- Heurística: `7 × 4 = 28` → próxima potência de 2 = **32**
- Potências de 2 são otimizadas para operações matriciais em GPU

**Por que D_FF = 64?**
- Convenção do transformer original: `D_FF = 4 × D_MODEL`
- Camada interna mais larga dá mais capacidade de transformação

---

### 00_overview.py — Mapa de orientação

**O que faz:** imprime um diagrama ASCII do fluxo completo interpolando valores de `config.py`. Não computa nada — é documentação viva.

---

### 01_tokenizer.py — Conversão de strings em números

**O que faz:** lê o CSV e converte cada evento em um array numérico `[5]`.

**Campos e transformações:**

| Campo | Tipo | Transformação |
|-------|------|---------------|
| `tipo` | categórico | `TIPOS["troca_senha"]` → `3` (índice inteiro) |
| `pais` | categórico | `PAISES["JP"]` → `3` |
| `device` | categórico | `DEVICES["disp_desc"]` → `3` |
| `valor` | contínuo | `valor / 10000.0` → `[0, 1]` |
| `hora` | contínuo | `hora / 23.0` → `[0, 1]` |

**Left-padding:** clientes com menos de 10 eventos têm as posições do início zeradas. Os eventos reais ficam nas últimas posições.

```
Cliente com 3 eventos (SEQ_LEN=10):
pos 0-6  → [0, 0, 0, 0, 0]   padding
pos 7    → login    BR  cel_hab   0.0  0.61
pos 8    → compra   BR  cel_hab   120  0.65
pos 9    → troca    JP  disp_desc 0.0  0.13
```

**Máscara:** `masks[cliente] = [0,0,0,0,0,0,0,1,1,1]` — indica quais posições são reais. Usada depois para ignorar padding na atenção.

**Saída:** `X [200,10,5]`, `y [200]`, `masks [200,10]`

---

### 02_embeddings.py — Números → vetores densos

**O que faz:** transforma cada campo numérico em um vetor de 32 dimensões.

**Dois mecanismos:**

| Campos | Mecanismo | Como funciona |
|--------|-----------|---------------|
| `tipo`, `pais`, `device` | `nn.Embedding` | Tabela `N×32`; o índice seleciona uma linha |
| `valor`, `hora` | `nn.Linear(1, 32)` | Multiplicação matricial: `y = W·x + b` |

**Por que embeddings para categóricos e linear para contínuos?**
- Índices inteiros não têm ordem matemática (3 não é "maior" que 1 semanticamente)
- Embeddings aprendem representações geométricas onde eventos similares ficam próximos
- Valores contínuos já têm escala — uma projeção linear basta

**Exemplo concreto para `troca_senha, JP, disp_desc, 0.0, 3h`:**
```
tipo   idx=3  → emb_tipo[3]   → [32 dims]
pais   idx=3  → emb_pais[3]   → [32 dims]
device idx=3  → emb_device[3] → [32 dims]
valor  0.000  → W · 0.0 + b   → [32 dims]
hora   0.130  → W · 0.13 + b  → [32 dims]
```

**Saída:** `[200, 10, 5, 32]` — 5 vetores de 32 por evento

---

### 03_local_fusion.py — 5 vetores → 1 token

**O que faz:** funde os 5 vetores de 32 dims em um único vetor de 32 dims que representa o evento completo.

**Por que isso importa?**
Antes da fusão, `troca_senha` e `dispositivo_desconhecido` são representações independentes. A fusão cria um vetor único onde a *combinação* dos campos pode ser aprendida.

**Mecanismo:**
```
concat([tipo, pais, device, valor, hora])  →  [160]   (5 × 32)
Linear(160, 32)                            →  [32]
ReLU + LayerNorm                           →  [32]  (estabilizado)
```

**Por que 160?** `D_MODEL × N_CAMPOS = 32 × 5 = 160`

**Como a Linear(160,32) captura combinações?**
Cada dimensão de saída faz um produto escalar com os 160 valores de entrada — ou seja, pode cruzar qualquer campo com qualquer outro. Os pesos são aprendidos durante o treino.

**Saída:** `[200, 10, 32]` — 1 token por evento

---

### 04_positional.py — Adição de posição

**O que faz:** soma um vetor de posição a cada token, para que o modelo saiba a ordem dos eventos.

**Por que é necessário?**
O transformer processa todos os tokens em paralelo — sem encoding de posição, o evento na posição 2 seria indistinguível do evento na posição 9.

**Fórmula (seno/cosseno):**
```
PE(pos, dim par)   = sin(pos / 10000^(dim/D_MODEL))
PE(pos, dim ímpar) = cos(pos / 10000^(dim/D_MODEL))
```

**Exemplo para pos=2:**
```
dim 0: sin(2.0 / 1.000) = sin(2.000) =  0.909
dim 1: cos(2.0 / 1.000) = cos(2.000) = -0.416
dim 2: sin(2.0 / 1.778) = sin(1.125) =  0.900
dim 3: cos(2.0 / 1.778) = cos(1.125) =  0.436
dim 8: sin(2.0 / 10.00) = sin(0.200) =  0.199
dim 9: cos(2.0 / 10.00) = cos(0.200) =  0.980
```

**Padrão das frequências:**
- Primeiras dimensões: oscilam rápido → distinguem posições próximas
- Últimas dimensões: oscilam devagar → distinguem posições distantes

**Por que não usar 0, 1, 2, 3...?**
- Escala: 9 seria muito maior que 0, distorcendo os embeddings
- Seno/cosseno têm padrão regular → modelo pode inferir posições não vistas

**Importante:** os valores de PE são **fixos** (`register_buffer`) — não são aprendidos, não têm gradiente.

**Saída:** `[200, 10, 32]` — shape igual, conteúdo enriquecido com posição

---

### 05_attention.py — Atenção scaled dot-product

**O que faz:** cada evento "olha" para todos os outros da sequência e decide o quanto cada um importa.

**Os três vetores:**

| Vetor | Pergunta | Gerado por |
|-------|----------|------------|
| Q (Query) | "o que estou procurando?" | `tokens @ Wq` |
| K (Key) | "o que ofereço aos outros?" | `tokens @ Wk` |
| V (Value) | "o que passo adiante se me escolherem?" | `tokens @ Wv` |

**Cálculo passo a passo:**
```
1. scores   = Q · K^T / √d_k        # similaridade entre todos os pares
2. scores   = masked_fill(pad, -1e9) # ignora padding
3. weights  = softmax(scores)        # normaliza → soma 1.0
4. output   = weights · V            # média ponderada dos valores
```

**Exemplo de pesos de atenção (último evento olhando para trás):**
```
pos 0-3  padding         → 0.00  (ignorado pela máscara)
pos 4    compra BR       → 0.02
pos 5    login BR        → 0.03
pos 6    troca_senha JP  → 0.31  ← mais relevante
pos 7    compra_negada   → 0.28
pos 8    compra JP       → 0.22
pos 9    transferencia   → 0.14
```

---

### 06_multihead.py — Multi-head attention

**O que faz:** roda o mecanismo de atenção 4 vezes em paralelo, cada cabeça com pesos independentes.

**Por que múltiplas cabeças?**
Cada cabeça pode aprender um tipo diferente de relação:
- Cabeça 1 → padrão temporal
- Cabeça 2 → padrão geográfico
- Cabeça 3 → padrão de valor
- Cabeça 4 → padrão de device

**Como funciona o split:**
```
[200, 10, 32]
  ↓ view
[200, 10, 4, 8]    # 4 cabeças × 8 dims cada (32/4)
  ↓ transpose
[200, 4, 10, 8]    # atenção roda em paralelo nas 4 cabeças
  ↓ atenção
[200, 4, 10, 8]
  ↓ transpose + view
[200, 10, 32]      # recombina
  ↓ Linear(32, 32) # Wo — aprende a misturar as cabeças
[200, 10, 32]
```

**Por que D_HEAD = 8?** `D_MODEL / N_HEADS = 32 / 4 = 8`
Convenção: D_MODEL sempre divisível por N_HEADS.

---

### 07_transformer.py — Backbone completo

**O que faz:** orquestra todos os blocos anteriores em um módulo coeso. Adiciona os sublayers padrão do transformer.

**TransformerEncoderLayer (nova peça):**
```
# Sublayer 1 — atenção com residual
att_out = MultiHeadAttention(x, mask)
x = LayerNorm(x + Dropout(att_out))

# Sublayer 2 — feed-forward com residual
ff_out = FeedForward(x)   # Linear(32,64) → ReLU → Linear(64,32)
x = LayerNorm(x + Dropout(ff_out))
```

**Dois conceitos críticos:**

**Residual connection** (`x + att_out`): soma a entrada original com a saída. Garante que o sinal não desapareça em redes profundas — mesmo que a atenção aprenda algo errado, o `x` original ainda passa intacto.

**LayerNorm:** normaliza os valores após cada sublayer para manter a escala estável durante o treino.

**Forward pass completo:**
```
Input          [200, 10, 5]
  ↓ embeddings [200, 10, 32]
  ↓ fusion     [200, 10, 32]
  ↓ positional [200, 10, 32]
  ↓ layer 1    [200, 10, 32]   ← atenção + feed-forward
  ↓ layer 2    [200, 10, 32]   ← padrões mais abstratos
Output         [200, 10, 32]
```

---

### 08_classifier.py — Score final

**O que faz:** pega o output do transformer e gera uma probabilidade de fraude `[0, 1]` por cliente.

**Por que só o último token?**
Após as camadas de atenção, o último token já "ouviu" todos os anteriores. Ele é o resumo contextualizado da sequência inteira.

**Mecanismo:**
```
last_token = output[batch, last_real_idx]    # [200, 32]
logit      = Linear(32, 1)(last_token)       # [200, 1]  sem limite de escala
score      = sigmoid(logit)                  # [200, 1]  ∈ [0, 1]
```

**Intuição do sigmoid:**
```
z = +3.0  →  sigmoid  →  0.95  →  provavelmente FRAUDE
z =  0.0  →  sigmoid  →  0.50  →  incerto
z = -3.0  →  sigmoid  →  0.05  →  provavelmente NORMAL
```

---

### 09_train_and_eval.py — Treino e avaliação

**O que faz:** treina o transformer, treina o XGBoost, e compara os dois.

**Loop de treino:**
```
for epoch in range(30):
    for batch in loader:
        scores = model(X, masks)        # forward pass
        loss   = criterion(scores, y)   # BCELoss com pesos
        loss.backward()                  # calcula gradientes de TODOS os pesos
        optimizer.step()                 # ajusta todos os pesos
```

**Onde os pesos são ajustados?** Aqui — `loss.backward()` propaga o gradiente por toda a rede: embeddings, fusion, atenção, classifier. Tudo de uma vez.

**Classe desbalanceada:** fraude = 30% dos dados. Solução:
```python
weights = torch.where(y == 1, pos_weight, 1.0)
# Errar num caso de fraude "pesa mais" no loss
```

**XGBoost vs Transformer:**

| | XGBoost | Transformer |
|--|---------|-------------|
| Entrada | Agregações por cliente (10 features) | Sequência completa (10 × 5) |
| Vê ordem? | Não | Sim |
| Vantagem | Robusto, treina rápido | Captura padrões temporais |

**Por que o transformer ganha em casos difíceis?**
O XGBoost sabe que houve uma `troca_senha`, mas não sabe que ela veio *depois* de um login normal e *antes* de uma transferência internacional às 3h. O transformer vê a sequência e aprende esse padrão.

**Interpretabilidade — pesos de atenção:**
```python
# Hook captura os pesos sem modificar o modelo
handle = model.transformer.layers[-1].attention.register_forward_hook(hook_fn)
```
Permite ver exatamente quais eventos o modelo priorizou para tomar a decisão.

---

## Conceitos-chave Consolidados

### Por que transformer para fraude?
Fraude quase nunca é um evento isolado — é uma *sequência*: login normal → troca de senha em país estrangeiro → transferência grande à madrugada. O transformer captura essa ordem; modelos clássicos como XGBoost não.

### O que é aprendido vs o que é fixo?

| Componente | Aprendido? |
|-----------|------------|
| Pesos de embedding (`emb_tipo`, etc.) | ✅ Sim |
| Pesos de `Linear(160, 32)` (fusion) | ✅ Sim |
| Positional encoding (seno/cosseno) | ❌ Fixo |
| Projeções Q, K, V da atenção | ✅ Sim |
| Pesos do classificador | ✅ Sim |

### Fluxo de dimensões resumido

```
[200, 10, 5]          → tokenizer (5 campos numéricos)
[200, 10, 5, 32]      → embeddings (cada campo virou vetor de 32)
[200, 10, 32]         → local fusion (5 vetores → 1 token)
[200, 10, 32]         → positional (+ informação de ordem)
[200, 10, 32]         → atenção × 2 camadas (+ contexto)
[200, 32]             → último token por cliente
[200, 1] → sigmoid    → probabilidade de fraude
```

### Heurísticas de design

| Decisão | Valor | Origem |
|---------|-------|--------|
| D_MODEL | 32 | `4 × max_categorias (7)` → 28 → 32 |
| D_FF | 64 | `4 × D_MODEL` (convenção transformer) |
| D_HEAD | 8 | `D_MODEL / N_HEADS = 32 / 4` |
| N_HEADS | 4 | Divisor de 32; cada cabeça aprende um padrão |
| N_LAYERS | 2 | Suficiente para dados pequenos sem overfitar |