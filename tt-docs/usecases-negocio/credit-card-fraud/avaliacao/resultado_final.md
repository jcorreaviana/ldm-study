# Resultado final — Transformer tabular para detecção de fraude

## Contexto

Dataset público de referência (Kaggle/ULB, cartões europeus, 2013) — 284.807 transações, 492 fraudes (0,173%), usado como estudo antes de replicar a abordagem nos dados reais do banco.

## Threshold operacional: 0,6226

Definido pela restrição de negócio: até 5% de falsos positivos entre transações legítimas (FPR ≤ 5%), maximizando recall no Cluster 1 (o perfil de fraude mais parecido com transação legítima, identificado via K-means). Falsos positivos disparam um segundo fator de autenticação — não bloqueio definitivo.

## Validação (usada para calibrar o threshold)

| Métrica | Valor |
|---|---|
| AUPRC | 0,7971 |
| Recall geral | 91,5% |
| Recall Cluster 1 | 89,8% (53/59) |
| FPR (legítimas) | 5,00% |
| Precisão | 3,0% |
| Matriz de confusão | TN=40.364, FP=2.124, FN=6, TP=65 |

## Teste (primeira e única vez, sem nenhum ajuste)

| Métrica | Valor |
|---|---|
| AUPRC | 0,7053 |
| Recall geral | 90,1% |
| Recall Cluster 1 | 86,0% (43/50) |
| FPR (legítimas) | 4,75% |
| Precisão | 3,1% |
| Matriz de confusão | TN=40.470, FP=2.018, FN=7, TP=64 |

## Generalização (validação → teste)

| Métrica | Diferença |
|---|---|
| AUPRC | −0,092 |
| Recall geral | −1,4pp |
| Recall Cluster 1 | −3,8pp |
| FPR | −0,25pp |
| Precisão | +0,1pp |

Recall e FPR generalizaram bem — diferenças pequenas, esperadas dado que o teste tem só 71 fraudes (amostra pequena, ruído estatístico natural). O AUPRC caiu mais (0,80→0,71); métrica mais sensível com poucos positivos, vale monitorar mas não superinterpretar isoladamente.

## Impacto financeiro (validação, valores do dataset público — não extrapolar diretamente para o R$ 8M/mês real do banco)

- Cluster 1: R$ 9.814 em risco → R$ 9.290 capturado (94,7%), R$ 524 perda residual.
- Todas as fraudes: R$ 10.637 em risco → R$ 10.113 capturado, R$ 524 perda residual (100% das perdas residuais vêm do Cluster 1 — fora dele, recall de 100% na validação).

## Conclusão

O modelo mantém o ponto de operação definido pelo negócio (FPR ≤ 5%, recall alto no Cluster 1) de forma consistente entre validação e teste. Para uso em produção real, os números de R$ e volume precisam ser recalibrados com os dados reais do banco — a arquitetura e a metodologia (limpeza sem vazamento, split estratificado, threshold calibrado só na validação, teste tocado uma única vez) são o que se transporta, não os valores absolutos deste dataset público.
