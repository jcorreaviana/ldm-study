# Limpeza dos dados e leitura de linearidade

## 1. Limpeza (`creditcard_clean.csv`)

Removidas apenas as 1.081 linhas duplicadas exatas (mantendo a primeira ocorrência). Não removi outliers estatísticos nas variáveis V, porque eles concentram a maior parte do sinal de fraude (visto na análise anterior) — remover teria apagado o próprio padrão que o modelo precisa aprender.

- Antes: 284.807 linhas, 492 fraudes
- Depois: 283.726 linhas, 473 fraudes (19 fraudes eram duplicatas)

Arquivo salvo: `creditcard_clean.csv`.

## 2. Linear ou não-linear?

**PCA (linear) vs t-SNE (não-linear)** nas 28 variáveis V, usando todas as fraudes + amostra de 5.000 legítimas (`viz_pca_vs_tsne.png`):

- Na projeção **linear (PCA)**, boa parte das fraudes já se separa em "braços" bem distintos da nuvem de transações legítimas — ou seja, existe um componente linear forte no problema.
- Na projeção **não-linear (t-SNE)**, aparece um cluster denso e bem isolado de fraudes (canto inferior esquerdo), mas também várias fraudes espalhadas dentro ou nas bordas dos clusters de transações legítimas. São esses casos "espalhados" que uma fronteira linear não resolve.

**Fronteira de decisão** nas duas variáveis mais discriminativas (V17 x V14), comparando um modelo linear com dois não-lineares (`viz_decision_boundary.png`):

| Modelo | AUC-ROC (5-fold CV) |
|---|---|
| Regressão Logística (linear) | 0,9494 |
| SVM kernel RBF (não-linear) | 0,9517 |
| Random Forest (não-linear) | 0,9534 |

A diferença entre linear e não-linear é pequena (menos de 0,5 ponto de AUC) quando se olha só para essas duas variáveis. A fronteira do Random Forest também aparece com um padrão "em blocos" — sinal de que parte da vantagem dele é ajuste a ruído local, não estrutura real.

## Conclusão prática

O problema **não é puramente linear**, mas também não é fortemente não-linear na maior parte dos casos: existe uma maioria de fraudes que uma fronteira reta já separa bem, e um núcleo menor de casos "difíceis", misturados com transações legítimas, onde interações entre variáveis importam mais. É exatamente para esse núcleo difícil que um mecanismo de atenção (capturando interações entre as V1-V28 como tokens de uma mesma transação, no formato tabular decidido) tende a agregar valor sobre um modelo linear — o ganho esperado é modesto, mas concentrado nos casos que mais importam (os que hoje escapam das regras simples).
