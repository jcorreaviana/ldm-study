# Tipos de Problemas de ML
> Guia de referência rápida para identificar o tipo de problema e escolher a abordagem correta.
> Construído a partir dos casos de uso estudados nos domínios financeiro, telecom e varejo.

---

## Como usar esse guia

Quando receber um problema de negócio, faça estas perguntas em ordem:

```
1. qual é a saída do modelo?
   → número contínuo, categoria, ranking, sequência?

2. tenho labels (respostas certas) nos dados de treino?
   → sim = supervisionado | não = não-supervisionado

3. o padrão depende de contexto histórico?
   → sim = sequencial | não = tabular

4. qual é o custo de cada tipo de erro?
   → define a métrica e o threshold
```

---

## 1. Classificação Binária

**Definição:** prever se algo pertence a uma de duas classes (sim/não, 0/1).

**Saída do modelo:** score entre 0 e 1 → aplicar threshold para decidir

**Quando usar:**
```
→  fraude ou não
→  churn ou não
→  inadimplência ou não
→  spam ou não
→  doença ou não
```

**Métricas:**
```
dataset balanceado:    AUROC, F1
dataset desbalanceado: AUPRC, Recall, Precision
nunca usar:            Accuracy (enganosa com desbalanceamento)
```

**Arquiteturas:**
```
tabular simples:    regressão logística, XGBoost
tabular complexo:   rede neural, FT-Transformer
sequencial:         transformer com self-attention, LSTM
```

**Exemplos nos nossos projetos:**
```
credit card fraud:  classificação binária tabular
                    AUPRC 0.705 com transformer tabular

PaySim fraud:       classificação binária tabular
                    AUPRC 0.9972 com XGBoost (baseline resolveu)

card testing:       classificação binária SEQUENCIAL
                    1 transação não detecta → transformer necessário
```

---

## 2. Classificação Multiclasse

**Definição:** prever a qual de N classes (> 2) um exemplo pertence.

**Saída do modelo:** vetor de probabilidades para cada classe (softmax)

**Quando usar:**
```
→  qual produto o cliente vai comprar? (A, B, C, D...)
→  qual categoria de fraude? (card testing, phishing, roubo de conta)
→  qual próximo evento do cliente? (compra, reclamação, cancelamento)
→  qual sentimento? (positivo, neutro, negativo)
→  qual próximo token? (LLMs — 50.000+ classes)
```

**Métricas:**
```
macro F1:         média entre todas as classes (trata todas igual)
weighted F1:      pondera pelo volume de cada classe
confusion matrix: ver quais classes confunde mais
```

**Arquiteturas:**
```
tabular:    XGBoost multiclasse, rede neural com softmax
texto:      BERT (classificação), GPT (geração)
sequencial: transformer com cabeça multiclasse
```

**Exemplo de aplicação NeoSpace:**
```
telecom:  classificar motivo de churn
          (preço, qualidade, atendimento, concorrência)
          permite ação de retenção específica por motivo
```

---

## 3. Regressão

**Definição:** prever um valor numérico contínuo.

**Saída do modelo:** número real (sem threshold)

**Quando usar:**
```
→  qual o valor do próximo pedido?
→  qual o score de crédito? (número, não apenas aprovado/reprovado)
→  qual o LTV (lifetime value) desse cliente?
→  quantos dias até o churn?
→  qual o preço justo desse imóvel?
```

**Métricas:**
```
MSE:   penaliza erros grandes mais que pequenos
MAE:   erro médio absoluto (mais interpretável)
R²:    % da variação explicada pelo modelo (0-1)
RMSE:  raiz do MSE (mesma unidade da variável alvo)
```

**Arquiteturas:**
```
linear:     regressão linear, ridge, lasso
não-linear: XGBoost regressor, rede neural com saída linear
sequencial: transformer com saída contínua
```

**Exemplo de aplicação NeoSpace:**
```
varejo:    prever o valor do próximo pedido de cada cliente
           permite personalizar ofertas por faixa de valor
           ex: cliente com LTV previsto de R$5.000 → oferta premium
```

---

## 4. Ranking / Recomendação

**Definição:** ordenar itens por relevância para um usuário específico.

**Saída do modelo:** score de afinidade para cada par (usuário, item)

**Quando usar:**
```
→  quais produtos recomendar para esse cliente?
→  quais ofertas mostrar primeiro?
→  quais clientes priorizar para abordagem comercial?
→  quais conteúdos são mais relevantes para esse usuário?
```

**Métricas:**
```
Precision@K:  dos K recomendados, quantos o usuário interagiu?
Recall@K:     de tudo que o usuário gostaria, quantos estão nos K?
NDCG:         mede se os mais relevantes aparecem no topo da lista
              (posição importa — 1º vale mais que 10º)
```

**Arquiteturas:**
```
colaborativo:  matrix factorization (usuário × item)
baseado em:    conteúdo → features do item
híbrido:       transformer com histórico do usuário (LDM)
```

**Exemplo nos nossos casos de uso:**
```
varejo (caso 3):  recomendação de produtos
                  catálogo de 2M produtos → top 10 por cliente
                  transformer com event stream do cliente
                  Precision@10, NDCG como métricas
```

---

## 5. Detecção de Anomalia

**Definição:** identificar exemplos que fogem do padrão normal — sem labels de treino.

**Saída do modelo:** score de anomalia (quanto esse exemplo é diferente do normal)

**Quando usar:**
```
→  detectar fraude quando não há labels históricos
→  detectar falha em equipamentos industriais
→  detectar comportamento incomum de usuário
→  detectar intrusão em redes (segurança)
```

**Diferença para classificação:**
```
classificação:   precisa de exemplos rotulados de fraude para treinar
anomalia:        treina só com exemplos normais
                 qualquer coisa diferente = suspeito
                 útil quando fraudes são novas/desconhecidas
```

**Métricas:**
```
AUROC:           separa normal de anômalo
threshold:       definido pelo percentil (ex: top 1% mais anômalos)
precision@K:     dos K mais anômalos, quantos são realmente anomalias?
```

**Arquiteturas:**
```
estatístico:    isolation forest, one-class SVM
autoencoder:    aprende a reconstruir o normal
                erro de reconstrução alto = anomalia
transformer:    aprende padrões normais, flagra desvios
```

---

## 6. Clustering (Agrupamento)

**Definição:** agrupar exemplos similares sem labels — descoberta de padrões.

**Saída do modelo:** rótulo de grupo para cada exemplo

**Quando usar:**
```
→  segmentar clientes por comportamento
→  identificar tipos de fraude (como fizemos com K-Means no credit card)
→  agrupar produtos similares
→  descobrir perfis de uso antes de criar modelos supervisionados
```

**Não é supervisionado — não tem resposta certa**

**Métricas:**
```
silhouette score:  quão bem separados estão os clusters
inertia:           soma das distâncias dentro de cada cluster
interpretação:     humano valida se os grupos fazem sentido de negócio
```

**Arquiteturas:**
```
K-Means:      agrupa por distância euclidiana (usado no credit card)
DBSCAN:       agrupa por densidade (bom para formas irregulares)
hierárquico:  constrói árvore de agrupamentos
```

**Exemplo nos nossos projetos:**
```
credit card:  K-Means nas 492 fraudes
              descobriu 3 tipos distintos de fraude
              Cluster 1 (77%) = maior prejuízo → prioridade do modelo
```

---

## 7. Previsão de Série Temporal

**Definição:** prever valores futuros com base em histórico temporal.

**Saída do modelo:** valor(es) no(s) próximo(s) instante(s) de tempo

**Quando usar:**
```
→  prever demanda de produtos (varejo)
→  prever volume de chamadas (telecom)
→  prever inadimplência no próximo mês (financeiro)
→  prever preço de ações
→  prever consumo de energia
```

**Diferença para regressão:**
```
regressão:        exemplos independentes entre si
série temporal:   cada exemplo depende dos anteriores
                  ordem importa — não pode embaralhar os dados
```

**Métricas:**
```
MAE:   erro médio absoluto por período
RMSE:  raiz do erro quadrático médio
MAPE:  erro percentual médio (interpretável)
```

**Arquiteturas:**
```
clássico:    ARIMA, Prophet (Facebook)
neural:      LSTM, GRU
transformer: Temporal Fusion Transformer, PatchTST
```

---

## 8. Geração de Texto (LLMs)

**Definição:** gerar texto coerente dado um contexto (prompt).

**Saída do modelo:** sequência de tokens (palavras/subpalavras)

**Quando usar:**
```
→  chatbots e assistentes virtuais
→  resumo automático de documentos
→  geração de relatórios
→  tradução automática
→  extração de informações de texto livre
```

**Arquiteturas:**
```
encoder-only:      BERT → compreensão (classificação, extração)
decoder-only:      GPT, Claude → geração (completar, conversar)
encoder-decoder:   T5, BART → tradução, resumo
```

---

## Resumo — Escolha Rápida

| Saída desejada | Tipo de problema | Modelo base |
|---|---|---|
| sim/não | classificação binária | XGBoost, transformer |
| qual categoria? | classificação multiclasse | XGBoost, BERT |
| qual valor? | regressão | XGBoost regressor, rede neural |
| qual ranking? | recomendação | transformer com event stream |
| isso é estranho? | anomalia | isolation forest, autoencoder |
| quais grupos? | clustering | K-Means, DBSCAN |
| qual próximo valor? | série temporal | Prophet, LSTM, transformer |
| qual próxima palavra? | geração | GPT, Claude |

---

## Quando o Transformer é Necessário

```
use transformer quando:
  ✓  padrão é sequencial (contexto histórico importa)
  ✓  interações entre features são complexas e não-lineares
  ✓  volume de dados é grande (> 100k exemplos)
  ✓  baseline tabular tem AUPRC < 0.70

não use transformer quando:
  ✗  baseline simples resolve (AUPRC > 0.90)
  ✗  padrão é transacional (1 evento com features certas detecta)
  ✗  poucos dados (< 10k exemplos)
  ✗  interpretabilidade é requisito obrigatório
```

---

## Relação com os Domínios da NeoSpace

| Domínio | Problemas comuns | Tipo |
|---|---|---|
| Financeiro | fraude, crédito, inadimplência | classificação binária |
| Financeiro | valor do próximo crédito | regressão |
| Telecom | churn, upsell | classificação binária |
| Telecom | previsão de demanda de rede | série temporal |
| Varejo | recomendação de produtos | ranking |
| Varejo | previsão de estoque | série temporal |
| Varejo | segmentação de clientes | clustering |
| Todos | detecção de padrões novos | anomalia |
EOF