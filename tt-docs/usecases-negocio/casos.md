# Casos de Uso — Tracking de Evolução
> Simulações de conversão de problema de negócio em solução de IA.
> Objetivo: preparação para entrevista NeoSpace — papel de conversão problema → solução.

---

## Framework de Avaliação

Cada caso de uso é avaliado em 5 pontos:

```
1. Problema central     →  identifica sintoma vs causa raiz + impacto financeiro
2. Tipo de problema ML  →  classificação, ranking, regressão, sequencial
3. Dados necessários    →  features relevantes, event stream, qualidade dos dados
4. Arquitetura          →  modelo adequado ao tipo de problema com justificativa
5. Métricas             →  técnicas + negócio + metodologia de validação (A/B)
```

**Escala:** 0 a 10 por caso de uso

---

## Caso 1 — Financeiro: Inadimplência em Fintech

**Data:** ~05/07/2026 (estimado)
**Domínio:** financeiro
**Score:** 7.5 / 10

### Contexto
Fintech de crédito com inadimplência subindo de 4% para 7%. Modelo atual usa score Serasa, renda declarada e histórico de pagamento. Data lake rico com histórico completo de eventos nunca utilizado.

### Respostas

**1. Problema central**
> Recomendação de crédito incorreta para clientes com perfil errado. O modelo não consegue discriminar bem os perfis de risco — aprova quem não deveria. Fatores externos (mercado, economia) também podem contribuir.

*Avaliação: correto em identificar a causa raiz. Faltou quantificar o impacto financeiro logo de início.*

**2. Tipo de problema**
> Classificação binária e não-linear. Múltiplas variáveis de naturezas diferentes (score, valor, data, device) com interações complexas.

*Avaliação: correto. Boa justificativa da não-linearidade.*

**3. Dados necessários**
> Explorar o data lake para identificar campos relevantes. Normalizar e classificar os dados (z-score, one-hot). Identificar features com mais peso. Correlações ocultas podem ser descobertas pelo modelo.

*Avaliação: correto. Mencionou correlações ocultas — ponto forte.*

**4. Arquitetura**
> Redes neurais profundas dado o volume e complexidade. Problema não-linear com múltiplos tipos de entrada.

*Avaliação: chegou na direção certa mas de forma hesitante. Faltou nomear transformer/LDM diretamente e justificar pela natureza sequencial dos dados.*

**5. Métricas**
> Matriz de confusão para ver os 4 cenários. Precision para saber quando o bloqueio foi correto. Recall para saber quantos inadimplentes foram capturados. A decisão entre os dois é de negócio.

*Avaliação: correto. Faltou mencionar AUPRC (dataset desbalanceado) e A/B test.*

### Oportunidades de Melhoria
- Quantificar impacto financeiro logo na pergunta 1
- Nomear transformer/LDM com mais confiança na pergunta 4
- Mencionar AUPRC para datasets desbalanceados
- Incluir A/B test na metodologia de validação

---

## Caso 2 — Telecom: Churn de Operadora

**Data:** ~06/07/2026 (estimado)
**Domínio:** telecom
**Score:** 8.0 / 10

### Contexto
Operadora com 8M de clientes perdendo 12% ao ano para concorrentes. Modelo atual usa apenas tempo de contrato, valor do plano e número de reclamações no último mês. Ação de retenção só acontece após o pedido de cancelamento — tarde demais.

### Respostas

**1. Problema central**
> Perda de carteira de clientes por concorrência. Falta de exploração dos dados disponíveis para diagnóstico correto. O modelo é reativo — age quando já é tarde. O dado rico existe mas não é usado.

*Avaliação: excelente. Separou sintoma de causa raiz. Identificou a natureza reativa como o problema central.*

**2. Tipo de problema**
> Classificação binária preditiva. A diferença para inadimplência: o evento ainda não aconteceu — precisa prever com antecedência (30/60/90 dias). Dataset desbalanceado: 88% não churna.

*Avaliação: correto e completo. Identificou a janela de predição e o desbalanceamento.*

**3. Dados necessários**
> Ligações para suporte, uso de dados por dia, variação de consumo, trocas de plano, NPS, reclamações com contexto temporal (não só contagem). Correlações ocultas: pagamento em atraso pode indicar insatisfação.

*Avaliação: excelente. Mencionou correlações ocultas e criticou agregações sem contexto temporal.*

**4. Arquitetura**
> Transformer com event stream. O problema é sequencial — o padrão de comportamento ao longo do tempo determina o risco. Médias destroem informação (mean tyranny). O mecanismo de atenção aprende quais eventos precedem o churn.

*Avaliação: resposta mais completa. Conectou com mean tyranny e justificou a sequencialidade.*

**5. Métricas**
> Recall prioritário (custo de um churner passando > custo de contatar cliente satisfeito). Impacto financeiro: 12% × 8M × ticket médio. Oportunidade de upsell além da retenção.

*Avaliação: excelente. Foi além do pedido identificando oportunidade de upsell. Faltou nomear AUPRC e A/B test formalmente.*

### Oportunidades de Melhoria
- Nomear AUPRC explicitamente para datasets desbalanceados
- Mencionar A/B test como metodologia de validação formal
- Na pergunta 4, nomear transformer com mais brevidade e confiança

---

## Caso 3 — Varejo: Recomendação de Produtos

**Data:** 07/07/2026
**Domínio:** varejo
**Score:** 8.5 / 10

### Contexto
Varejista online com 15M clientes e 2M produtos. Taxa de conversão das recomendações em 1.2% vs meta de 3%. Sistema atual usa apenas 3 últimas compras. Clientes com recomendações relevantes gastam 4x mais.

### Respostas

**1. Problema central**
> Perda de oportunidade de receita por recomendações imprecisas. Causa: modelo usa features pobres (3 compras) ignorando o contexto comportamental completo. Data lake rico inexplorado.

*Avaliação: excelente. Identificou causa raiz e quantificou o impacto.*

**2. Tipo de problema**
> Recomendação por ranking. O modelo calcula score de afinidade para cada par (cliente, produto) e ordena — recomenda os top N.

*Avaliação: correto. Chegou em "ranking" com mais precisão que nas sessões anteriores.*

**3. Dados necessários**
> Compras realizadas, carrinhos abandonados, buscas, avaliações, categorias e marcas favoritas, faixa de valor. Insight adicional: não só o quê recomendar mas como apresentar (desconto, frete, bundle).

*Avaliação: excelente. Insight prescritivo espontâneo foi o destaque da sessão.*

**4. Arquitetura**
> Transformer com mecanismo de atenção sobre o event stream. A sequência temporal importa — carrinho abandonado recente vale mais que compra de 6 meses em categoria diferente. Score de afinidade ordena os 2M produtos.

*Avaliação: resposta mais direta e confiante das três sessões.*

**5. Métricas**
> Taxa de conversão (1.2% → 3%), carrinhos finalizados, avaliação dos produtos recomendados, ticket médio. A/B test: grupo com modelo vs sem modelo por 30 dias.

*Avaliação: excelente. A/B test incluído naturalmente. Faltou Precision@K e NDCG — métricas padrão de ranking.*

### Oportunidades de Melhoria
- Nomear Precision@K e NDCG — métricas específicas de sistemas de recomendação
- Na pergunta 2, ir direto para "ranking" sem passar por classificação primeiro

---

## Evolução dos Scores ao Longo do Tempo

| Data | Caso | Domínio | Score |
|---|---|---|---|
| ~05/07/2026 | Inadimplência em Fintech | Financeiro | 7.5 / 10 |
| ~06/07/2026 | Churn de Operadora | Telecom | 8.0 / 10 |
| 07/07/2026 | Recomendação de Produtos | Varejo | 8.5 / 10 |

```
caso 1 — financeiro:   7.5 / 10
caso 2 — telecom:      8.0 / 10
caso 3 — varejo:       8.5 / 10
```

```
progresso por dimensão:

                    caso 1    caso 2    caso 3
problema central      ★★★★      ★★★★★     ★★★★★
tipo de problema      ★★★★      ★★★★★     ★★★★★
dados necessários     ★★★★      ★★★★★     ★★★★★
arquitetura           ★★★       ★★★★      ★★★★★
métricas              ★★★★      ★★★★      ★★★★
```

---

## Padrões de Melhoria Identificados

**Consistentemente bom:**
- Identificar causa raiz além do sintoma
- Conectar dados disponíveis com o problema
- Visão de negócio além da técnica

**Em evolução:**
- Nomear a arquitetura com mais confiança e brevidade
- Quantificar impacto financeiro desde o início

**A consolidar:**
- Métricas técnicas específicas por tipo de problema:
  - classificação desbalanceada → AUPRC
  - ranking → Precision@K, NDCG
  - regressão → R², MAE
- A/B test como validação padrão em todos os casos

---

## Próximos Casos de Uso

```
⬜  financeiro 2  →  detecção de fraude em tempo real
⬜  telecom 2     →  upsell e upgrade de plano
⬜  varejo 2      →  previsão de demanda / estoque
⬜  cross-domain  →  problema misto (ex: fintech + varejo)
```

---

## Glossário de Métricas por Tipo de Problema

| Problema | Métricas Técnicas | Métricas de Negócio |
|---|---|---|
| Classificação binária balanceada | AUROC, F1 | taxa de acerto, redução de erro |
| Classificação binária desbalanceada | AUPRC, Recall, Precision | custo do erro tipo I vs tipo II |
| Ranking / Recomendação | Precision@K, NDCG, Recall@K | taxa de conversão, CTR |
| Regressão | R², MAE, RMSE | erro médio em unidade de negócio |
| Sequencial / Temporal | AUROC + gap temporal | antecedência da predição |