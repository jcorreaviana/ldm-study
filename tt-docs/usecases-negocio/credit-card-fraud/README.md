# Notas do Desafio — Detecção de Fraude com Transformer Tabular
> Registro das decisões, análises e aprendizados do projeto prático.
> Dataset: Credit Card Fraud Detection (Kaggle — mlg-ulb)
> Formato: par híbrido — José (negócio + validação) + Claude/Cowork (implementação)

---

## Estrutura do Projeto

```
credit-card-fraud/
│
├── eda/
│   ├── EDA_creditcard.md          →  análise exploratória completa
│   ├── viz_correlation.png        →  matriz de correlação entre features
│   ├── viz_decision_boundary.png  →  fronteira de decisão: linear vs não-linear
│   ├── viz_pca_vs_tsne.png        →  separabilidade das 28 features (PCA e t-SNE)
│   └── viz_kmeans_clusters.png    →  3 tipos de fraude identificados via K-Means
│
├── preprocessing/
│   ├── preprocess_pipeline.py     →  script completo de pré-processamento
│   ├── preprocessing_meta.json    →  estatísticas do treino (médias, desvios, splits)
│   └── limpeza_e_linearidade.md   →  decisões e justificativas do pré-processamento
│
├── modelo/
│   ├── transformer_fraude.py      →  arquitetura FT-Transformer implementada
│   └── checkpoint_melhor_modelo.pt →  pesos do melhor modelo (época 27, AUPRC 0.797)
│
├── avaliacao/
│   ├── curva_pr.png               →  curva Precision × Recall com threshold marcado
│   ├── matriz_confusao.png        →  matriz de confusão no conjunto de teste
│   └── resultado_final.md         →  métricas completas validação vs teste
│
└── README.md                      →  este arquivo
```

---

## Como Reproduzir

```bash
# 1. pré-processamento
python preprocessing/preprocess_pipeline.py

# 2. treino (retoma do checkpoint se existir)
python modelo/transformer_fraude.py

# 3. avaliação com threshold fixo
# threshold 0.6226 — definido na validação, aplicado uma única vez no teste
```

---

## Contexto de Negócio

**Cliente simulado:** banco digital brasileiro com 2 milhões de cartões ativos

**Problema:**
- Chargebacks por fraude subiram 40% nos últimos 6 meses
- Custo atual: R$8M/mês em perdas
- Modelo atual: regras manuais — bloqueia Amount > R$5.000 ou países não autorizados
- Consequências: bloqueia clientes legítimos e deixa passar fraudes sofisticadas

**Sobre o dataset:**
O `creditcard.csv` (284.807 transações, 492 fraudes) é o dataset público de referência do Kaggle/ULB (cartões europeus, 2013), usado aqui como estudo. As variáveis V1-V28 são componentes de PCA sobre dados originais nunca divulgados. Isso significa:
- Os valores em R$ não representam a operação real do banco
- Não dá para traduzir os padrões aprendidos em regras de negócio interpretáveis sem os dados originais
- O que se transporta para produção é a **metodologia**, não os números

---

## Etapa 1 — EDA (Análise Exploratória)

### Achados principais

**Desbalanceamento extremo:**
```
492 fraudes em 284.807 transações = 0.17%  (ratio 578:1)
→  accuracy não serve como métrica
→  métrica correta: AUPRC
```

**Qualidade dos dados:**
```
1.081 linhas duplicadas (19 são fraude)
→  risco de data leakage — removidas antes do split
```

**Valor da transação (Amount):**
```
fraude: mediana R$9,25  |  legítima: mediana R$22,00
27 casos com Amount=0   →  card testing
Amount × Class = 0.01   →  valor sozinho NÃO discrimina fraude
→  regra atual (Amount > R$5.000) usa a feature com MENOR poder discriminativo
```

**Padrão horário:**
```
taxa de fraude geral: 0.17%
taxa de fraude às 2h: 1.71%  (10x maior)
→  sinal forte e independente das features V
```

**Features mais discriminativas:**
```
8 features com Cohen's d entre 3.8 e 8.3
V17, V14, V12, V10, V16, V3, V7, V11
todas ortogonais entre si (PCA garante)
→  cada feature captura padrão diferente de fraude
→  combinação delas é o que detecta fraude, não cada uma isolada
```

### Matriz de Correlação

```
features V entre si:   ~0.00  →  ortogonais (PCA garante)
features V vs Class:   -0.33 a +0.15  →  correlações fracas a moderadas
Amount vs Class:        0.01  →  praticamente zero
V7 × Amount:            0.40  →  única correlação relevante entre features
```

**Conclusão:** nenhuma feature sozinha detecta fraude. Problema não-linear confirmado.

### Análise de Separabilidade

**Fronteira de decisão (V17 × V14):**
```
regressão logística:  fronteira linear não separa — alto FN
SVM kernel RBF:       separação melhor, ainda tem hard cases
Random Forest:        fronteira em blocos — risco de overfitting
→  problema não-linear confirmado
→  regra "V17 < -2 E V14 < -2" reduziria FP mas aumentaria FN
   fraudes sofisticadas próximas ao zero passariam
```

**PCA 2D vs t-SNE:**
```
PCA 2D (linear):   3 clusters distintos visíveis, mistura com legítimas
t-SNE (não-linear): separação muito melhor — estrutura interna confirmada
→  fraudes não são aleatórias — têm sub-padrões
→  multi-head attention justificado
```

### K-Means (K=3) nas Fraudes

| Cluster | Volume | Amount médio | Perfil |
|---|---|---|---|
| 0 | ~21% | R$86,59 | padrão claro, bem separado — fácil de detectar |
| 1 | ~77% | R$136,54 | majoritário e sutil — maior desafio e maior prejuízo |
| 2 | ~1.5% | R$1,08 | card testing — precursor de fraude maior |

**Análise financeira:**
```
Cluster 1:  365 × R$136,54 = R$49.837 em risco  ← prioridade de negócio
Cluster 0:  101 × R$86,59  = R$8.745 em risco
Cluster 2:    7 × R$1,08   = R$7,56 em risco
```

**Insight:** Cluster 2 (card testing, R$1,08) é precursor dos clusters 0 e 1. Com event stream por cliente, o transformer aprenderia a detectar card testing ANTES da fraude principal — ação preventiva. Sem clienteID, esse sinal sequencial é perdido.

### Decisão de Arquitetura

```
dataset não tem clienteID
→  não dá para montar event stream por cliente
→  transformer com atenção ENTRE FEATURES (não entre transações)
→  similar ao TabNet — mais honesto com os dados disponíveis
→  para LDM real com event stream: migrar para PaySim
```

---

## Etapa 2 — Pré-processamento

```
ordem executada:
1. remoção de duplicatas (antes do split — evita data leakage por linha)
2. split estratificado 70/15/15
3. z-score em Amount e Time (fit só no treino — evita data leakage por estatística)
   V1-V28 já normalizados pelo PCA
4. class_weight balanceado: {0: 0.50, 1: 300.01}
   decisão: não ajustar por cluster
   motivo: clusters 0 e 2 já são naturalmente separados
   o peso de 300x beneficia principalmente o Cluster 1

K-Means refeito só com as 331 fraudes do treino (não as 473 originais)
→  evita data leakage via clustering
→  cluster nunca vira feature de entrada (seria circular)
```

**Resultado do split:**
| Conjunto | Linhas | Fraudes | % fraude |
|---|---|---|---|
| Treino (70%) | 198.608 | 331 | 0.1667% |
| Validação (15%) | 42.559 | 71 | 0.1668% |
| Teste (15%) | 42.559 | 71 | 0.1668% |

Sobreposição entre conjuntos: zero.

---

## Etapa 3 — Arquitetura do Transformer

### Correções importantes durante o design

**Correção 1 — cabeças de atenção:**
```
instrução original:  "3 cabeças = 1 por cluster"
correto:             cabeças se especializam sozinhas durante o treino
                     a prioridade do Cluster 1 entra na função de perda
                     não na arquitetura
```

**Correção 2 — "regressão logística falha":**
```
gap real:  AUC linear 0.9494 vs não-linear 0.9534  (diferença pequena)
argumento correto:
  transformer captura padrões COMBINADOS em múltiplas features
  exatamente o perfil do Cluster 1 (desvios pequenos espalhados em várias V)
  não que linear "falha" — mas que não captura esse tipo específico de padrão
```

### Arquitetura implementada (FT-Transformer)

```
entrada:    30 features (V1-V28 + Amount_z + Time_z)
            cada feature vira token via projeção linear própria
            + token [CLS] para classificação
            ↓
3 blocos encoder:
  4 cabeças de atenção entre features
  dimensão 64
            ↓
[CLS] final → classificador → logit → sigmoid
total:      106.561 parâmetros
```

### Treino

```
30 épocas (checkpoints incrementais — compute lento em CPU ~80s/época)
melhor AUPRC de validação: 0.7971 na época 27
early stopping por patience
```

---

## Etapa 4 — Escolha do Threshold

### Tensão identificada entre dois critérios de negócio

```
critério A — "precisão ≥ 95%"
  threshold: 0.9998
  recall C1: 32%   ← muito baixo
  FPR:       ~0%   ← quase zero FP

critério B — "até 5% de FPR entre legítimas"
  threshold: 0.6226
  recall C1: 89.8% ← alto ✓
  FPR:       5.00% ← no limite ✓
```

**Decisão:** critério B — justificado pelo uso de segundo fator de autenticação em vez de bloqueio definitivo. Vale checar esse tipo de contradição antes de fixar um threshold — as duas leituras levam a operações completamente diferentes.

---

## Etapa 5 — Avaliação Final

### Resultado no teste (threshold 0.6226, tocado uma única vez)

| Métrica | Validação | Teste | Diferença |
|---|---|---|---|
| AUPRC | 0.7971 | 0.7053 | -0.092 |
| Recall geral | 91.5% | 90.1% | -1.4pp |
| Recall Cluster 1 | 89.8% | 86.0% | -3.8pp |
| FPR (legítimas) | 5.00% | 4.75% | -0.25pp |
| Precisão | 3.0% | 3.1% | +0.1pp |

**Matriz de confusão (teste):**
```
TN=40.470  FP=2.018
FN=7       TP=64
```

**O modelo generalizou?**
```
recall e FPR:  sim — diferenças dentro do ruído estatístico (71 fraudes de teste)
AUPRC:         queda maior (-0.092) — métrica sensível com poucos positivos
               monitorar sem superinterpretar isoladamente
               o que importa para o negócio é o ponto de operação:
               recall 86% + FPR 4.75% = critérios mantidos ✓
```

---

## Síntese para o Cliente

**Problema identificado:**
```
3 tipos distintos de fraude — Cluster 1 é 77% das fraudes e R$49.837 em risco
regra atual (Amount > R$5.000) não captura esse cluster (mediana R$136)
```

**Resultado do modelo:**
```
recall Cluster 1:  86%    (43 de 50 fraudes capturadas no teste)
FPR:               4.75%  (segundo fator, não bloqueio definitivo)
proteção:          ~R$42.000 de R$49.837 em risco
```

**Ações operacionais:**
```
score > 0.95:     bloqueio automático
score 0.62-0.95:  segundo fator de autenticação
score < 0.62:     libera normalmente
mesa de operação: monitorar casos de segundo fator
```

**Próximos passos:**
```
1. recalibrar threshold com dados reais do banco (volume, R$, MCCs)
2. investigar se campos não anonimizados explicam o Cluster 1
   → pode virar regra explicável complementar ao modelo
3. shadow mode 30-60 dias com dados reais
4. gradual rollout por perfil de cliente
5. reavaliar fator de boost do Cluster 1 na loss (usado 2x sem tuning extensivo)
6. evolução para LDM com event stream por cliente (dataset atual sem clienteID)
```

---

## Score Final do Desafio: 8.5 / 10

### O que foi muito bom
```
✓  identificou 3 tipos distintos de fraude e priorizou pelo impacto financeiro
✓  conectou card testing como precursor de fraude maior
✓  decisão correta de class_weight (não ajustar por cluster)
✓  proposta prescritiva (segundo fator vs bloqueio definitivo)
✓  ordem correta de leitura: recall C1 → FPR → AUPRC
✓  identificou ambiguidade entre precisão ≥ 95% e 5% FPR
✓  propôs shadow mode e gradual rollout corretamente
✓  reconheceu limitação do dataset (sem clienteID → sem event stream)
✓  aceitou e incorporou correções técnicas do Cowork
```

### Oportunidades de melhoria
```
→  calibrar afirmações técnicas com evidência
   ("regressão logística falha" vs "transformer captura padrões combinados melhor")

→  cabeças de atenção se especializam sozinhas
   controle via função de perda, não via número de cabeças

→  quantificar impacto financeiro mais cedo na análise
   (R$49.837 deveria ter aparecido na EDA, não só no final)

→  nomear métricas formais com mais naturalidade
   Precision@K, AUPRC, FPR — ainda hesita em alguns momentos
```

### Comparativo com casos de uso anteriores
| Data | Caso | Domínio | Score |
|---|---|---|---|
| ~05/07/2026 | Inadimplência em Fintech | Financeiro | 7.5/10 |
| ~06/07/2026 | Churn de Operadora | Telecom | 8.0/10 |
| 07/07/2026 | Recomendação de Produtos | Varejo | 8.5/10 |
| 10/07/2026 | Detecção de Fraude (projeto) | Financeiro | 8.5/10 |