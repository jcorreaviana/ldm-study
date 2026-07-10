# Análise exploratória — dataset de transações (creditcard.csv)

## Alerta importante antes de tudo

A estrutura deste arquivo (Time, V1–V28, Amount, Class, 284.807 linhas, 492 casos positivos, 2 dias de dados) corresponde exatamente ao dataset público "Credit Card Fraud Detection" (Kaggle/ULB, 2013), onde V1–V28 são componentes de PCA aplicados sobre variáveis originais que nunca foram divulgadas por sigilo. Duas implicações práticas:

1. Se este for de fato o dataset público (e não uma extração real do seu processamento com a mesma nomenclatura), os números de valor de fraude e o volume de 2 dias **não representam necessariamente a operação do banco** — vale confirmar a origem antes de usar para dimensionar o problema de R$ 8M/mês.
2. As colunas V1–V28 são combinações lineares anônimas de variáveis originais. Isso significa que dá para treinar um modelo preditivo em cima delas, mas **não dá para traduzir os padrões encontrados em regras de negócio interpretáveis** (tipo "declinar se MCC=X e horário=Y") sem acesso às variáveis originais do seu processamento. Isso é relevante para comparar com as regras manuais atuais, que presumivelmente são baseadas em campos de negócio (MCC, geolocalização, canal, etc.), não em componentes de PCA.

Independentemente disso, o que segue é o que o arquivo, tal como está, mostra.

## 1. Estrutura geral

- 284.807 transações, 31 colunas, todas numéricas, sem valores nulos.
- 1.081 linhas duplicadas exatas (19 delas com Class=1). Duplicatas de fraude infladas artificialmente podem vazar entre treino/teste e superestimar a performance de um modelo — recomendo `drop_duplicates()` antes de qualquer split.
- `Time` cobre 172.792 segundos (≈48h) contínuos, consistente com "últimos 2 dias".

## 2. Desbalanceamento de classes

- 492 fraudes em 284.807 transações → **0,173%**, proporção de **578 transações legítimas para cada fraude**.
- Esse desbalanceamento é o ponto mais crítico para qualquer modelagem: accuracy não serve como métrica (um modelo que nunca aponta fraude já acerta 99,8%). É preciso usar precision/recall, PR-AUC, e técnicas como class weighting, undersampling/oversampling (SMOTE) ou thresholds ajustados a partir da curva precision-recall.

## 3. Valor da transação (Amount)

- Mediana de fraude é bem menor que a legítima (R$ 9,25 vs R$ 22,00) — contraintuitivo, sugere que fraudadores testam com valores pequenos antes de golpes maiores ("card testing").
- 27 fraudes com Amount = 0 — reforça a hipótese de teste de cartão.
- O maior valor fraudulento é R$ 2.125,87, enquanto a maior transação legítima chega a R$ 25.691,16 — fraude não está concentrada nos valores mais altos.

## 4. Padrão temporal

- Fraude tem taxa de ~0,17% na média, mas dispara para **1,71% às 2h da manhã** e **1,04% às 4h** — horários de menor volume de transações legítimas. Esse é um sinal temporal forte e barato de explorar (hora do dia sozinha já discrimina risco).

## 5. Features V1–V28: sinal muito forte

Calculei Cohen's d (diferença de médias normalizada) entre fraude e legítima para cada variável. Oito delas têm separação muito acima do que normalmente se vê em dados de fraude reais:

| Feature | Cohen's d | Correlação com Class |
|---|---|---|
| V17 | -8.32 | -0.33 |
| V14 | -7.64 | -0.30 |
| V12 | -6.50 | -0.26 |
| V10 | -5.35 | -0.22 |
| V16 | -4.83 | -0.20 |
| V3  | -4.74 | -0.19 |
| V7  | -4.59 | -0.19 |
| V11 |  3.78 |  0.15 |

Para referência, um Cohen's d acima de 2 já é considerado separação "enorme" em ciências sociais/negócio — aqui há 5 variáveis acima de 4,7. Isso indica que um modelo supervisionado simples deve superar substancialmente regras de threshold único, porque o sinal de fraude está distribuído em várias dimensões ortogonais entre si (confirmado no heatmap de correlação: essas variáveis têm correlação ~0 entre si, por serem componentes de PCA).

## 6. Conclusão da fase de exploração

- Os dados, como estão, **carregam sinal muito forte e viável para modelagem supervisionada** — a separação de classes em várias features é bem acima do que normalmente se observa. Isso é uma faca de dois gumes: se este for realmente o dataset de produção, ótimo; se for o dataset público de referência, os resultados de um modelo aqui **não devem ser extrapolados diretamente** para a taxa real de detecção esperada em produção.
- O maior risco técnico não é falta de sinal, é o desbalanceamento (578:1) e a duplicação de registros — ambos exigem tratamento antes de treinar.
- Regras manuais que usam poucos campos (ex.: valor + horário) plausivelmente perdem combinações capturadas por essas 8 variáveis ortogonais — é o argumento mais forte a favor de um modelo.

Antes de eu montar qualquer modelo, faz sentido confirmar: (1) se este dataset é realmente o de produção do banco ou uma referência pública, e (2) o que as regras manuais atuais usam hoje, para comparar de forma justa contra um modelo.
