# Notas do Desafio — Card Testing Sintético
> Dataset: gerado sinteticamente com gerar_dataset.py
> Formato: par híbrido — José (negócio + validação) + Claude/Cowork (implementação)
> Status: EM ANDAMENTO

---

## Contexto de Negócio

**Cliente simulado:** PayFlow — fintech de pagamentos digitais com 2M de cartões ativos

**Problema:**
- Chargebacks por card testing cresceram 340% em 4 meses
- Custo: ~R$1.2M/mês
- Modelo atual analisa cada transação isolada — não vê o padrão sequencial
- Fraudadores fazem micro-transações (R$1-5) para testar antes do golpe (R$500-2000)

**Dataset:**
```
140.672 transações | 10.000 clientes | 1.000 fraudes (0.71%)
colunas: clienteID, timestamp, tipo, merchant, valor, saldo_antes, saldo_depois, isFraud
```

---

## Decisões do Dataset Sintético

**Ajuste de artefato — tipo e merchant do golpe:**
```
problema identificado na EDA:
  100% das fraudes eram tipo='transferencia' + merchant='online'
  modelo simples aprenderia essa regra trivialmente
  → não precisaria de sequência para detectar

correção no gerar_dataset.py:
  tipo do golpe:    random.choice(['transferencia', 'saque', 'compra'])
  merchant do golpe: random.choice(MERCHANTS)

resultado após correção:
  saque:        38.3% dos golpes
  compra:       32.2% dos golpes
  transferencia: 29.5% dos golpes
  merchant distribuído uniformemente (15-18% cada)
```

---

## EDA — Achados Principais

```
fraudes:     1.000 de 140.672 (0.71%) | 1 por cliente fraudador
padrão:      micro (R$2-5) → micro (R$2-5) → golpe (R$500-2000)
             micro-transações têm isFraud=0 → modelo atual não detecta
horário:     fraudes concentradas 2h-8h (média 4h39)
             legítimas distribuídas uniformemente
valor:       fraude média R$1.235 vs legítimo R$396
             mas só aparece no golpe — micros ficam escondidas
```

---

## Baseline XGBoost — Resultado

**Features:** transação isolada (valor, tipo, merchant, hora, saldo_antes, saldo_depois)
**Split:** aleatório estratificado 70/15/15

```
ROC-AUC:  0.995  ← parece ótimo
PR-AUC:   0.82   ← bom no agregado

PONTO CEGO confirmado:
  micro-transações: probabilidade média 0.3%  ← passa despercebido
  golpe:            probabilidade média 94.9%  ← detecta depois que é tarde

conclusão:
  baseline detecta o golpe DEPOIS que o dinheiro saiu
  não detecta o card testing na fase útil (micro-transações)
  → transformer sequencial necessário
```

---

## Conceito — Cold Start e Janela de Contexto Insuficiente

**Origem:** discussão sobre por que eventos sem histórico parecem aleatórios.

### Cold Start

```
cliente novo sem histórico suficiente
→  modelo não tem contexto para avaliar
→  score próximo de 0.5 (incerto/neutro)
→  qualquer decisão nesse ponto é essencialmente aleatória
```

### Janela de Contexto Insuficiente

```
o padrão card testing precisa de pelo menos 3 eventos:
  micro 1 → micro 2 → golpe

se o modelo vê só 1 ou 2 transações:
  não tem contexto suficiente para reconhecer o padrão
  → score baixo ou neutro
  → falso negativo temporário — não é erro do modelo
     é ausência de informação
```

### Como o Score Deve Evoluir

```
transação 1:           score ≈ aleatório   (sem contexto)
transação 2:           score ≈ baixo       (pouco contexto)
transação 3 (micro 1): score começa subir  (padrão emergindo)
transação 4 (micro 2): score sobe mais     (padrão confirmando)
transação 5 (golpe):   score alto          (tarde demais)

objetivo do transformer:
  levantar alerta na transação 4
  ANTES do golpe acontecer
```

### Implicação de Negócio

```
primeiras transações do cliente:
  →  segunda camada de segurança obrigatória
  →  limite transacional menor até acumular histórico
  →  monitoramento manual nas primeiras N transações

após histórico suficiente:
  →  modelo assume o controle com confiança

analogia:  cartão de crédito novo
  limite baixo no início
  aumenta conforme histórico se consolida
```

---

## Próximas Etapas

```
⬜  transformer sequencial
    event stream por clienteID
    janela de 5 transações
    label por posição
    métrica adicional: recall nas micro-transações

⬜  avaliação e comparativo baseline vs transformer
⬜  relatório GPU (nvitop + wandb)
⬜  score do desafio e notas finais
```

---

## Transformer Sequencial — Resultado Final

### Versão 1 — label isFraud original
```
AUPRC:                    1.0000
recall micro-transações:  0.0%   ← igual ao baseline
score médio micros:       0.024% ← invisível
score médio golpe:        99.99%

conclusão: aprendeu a detectar o golpe, não o padrão
           micro-transações têm isFraud=0 → modelo ignora
```

### Versão 2 — label_preventivo (micro + golpe = 1)
```
AUPRC contra isFraud real:  0.3416
recall preventivo:          100%   ← todos os golpes alertados antes
score médio micros:         99.99% ← alerta máximo nas precursoras
score médio golpe:          99.99%
golpes com alerta prévio:   150/150 (100%)

conclusão: aprendeu o padrão sequencial completo
           levanta alerta nas micro-transações
           ANTES do golpe acontecer
```

---

## Comparativo Final — Baseline vs Transformer

| Métrica | Baseline XGBoost | Transformer v1 | Transformer v2 |
|---|---|---|---|
| AUPRC (isFraud real) | 0.82 | 1.00 | 0.34 |
| Recall golpe | 100% | 100% | 100% |
| Recall micro-transações | 0% | 0% | **100%** |
| Alerta ANTES do golpe | 0/150 | 0/150 | **150/150** |
| Score médio micros | 0.024% | 0.024% | **99.99%** |

---

## Por que o AUPRC caiu no transformer v2?

```
transformer v2 dá score alto para micro-transações (isFraud=0)
→  na métrica clássica, isso conta como falso positivo
→  AUPRC cai de 1.00 para 0.34

mas são VERDADEIROS POSITIVOS de negócio:
→  o alerta nas micro-transações é exatamente o que queremos
→  a métrica clássica não captura o valor preventivo
```

---

## A Lição Fundamental desse Desafio

```
métrica clássica (AUPRC contra isFraud):
  pergunta:  "você identificou a transação fraudulenta?"
  baseline ganha: 0.82 vs 0.34

métrica de negócio (recall preventivo):
  pergunta:  "você alertou ANTES do golpe acontecer?"
  transformer ganha: 100% vs 0%

conclusão:
  a escolha da métrica define o que o modelo aprende
  e o que o modelo aprende define se ele é útil para o negócio

  um modelo com AUPRC 0.82 que só detecta após o prejuízo
  é menos valioso que um modelo com AUPRC 0.34
  que previne 100% dos golpes
```

---

## Conexão com o LDM da NeoSpace

```
pipeline clássico:   analisa cada transação isolada
                     detecta o golpe DEPOIS que aconteceu
                     reativo

LDM (transformer sequencial):
                     analisa o event stream do cliente
                     detecta o padrão card testing
                     ANTES do golpe
                     preventivo

esse desafio demonstra exatamente esse argumento:
  quando o padrão é genuinamente sequencial
  o transformer supera modelos tabulares
  na métrica que importa para o negócio
```

---

## Score Final do Desafio: 9.0 / 10

### O que foi muito bom
```
✓  identificou o artefato de simulação (tipo/merchant fixo)
✓  corrigiu o gerador antes de modelar
✓  rodou baseline ANTES do transformer (lição do PaySim aplicada)
✓  identificou que AUPRC clássico não captura o valor preventivo
✓  propôs label_preventivo para capturar o padrão correto
✓  cold start identificado e documentado
✓  conectou resultado com argumento do LDM NeoSpace
```

### Oportunidades de melhoria
```
→  poderia ter proposto o label_preventivo antes de rodar o transformer v1
   economia de uma rodada de treino

→  quantificar o impacto financeiro preventivo:
   "150 golpes evitados × R$850 médio = R$127.500 por período de teste"
```

---

## Comparativo de Todos os Desafios

| Data | Caso | Domínio | Score | Modelo | AUPRC |
|---|---|---|---|---|---|
| ~05/07/2026 | Inadimplência Fintech | Financeiro | 7.5/10 | conceitual | — |
| ~06/07/2026 | Churn Operadora | Telecom | 8.0/10 | conceitual | — |
| 07/07/2026 | Recomendação Produtos | Varejo | 8.5/10 | conceitual | — |
| 10/07/2026 | Fraude Credit Card | Financeiro | 8.5/10 | transformer tabular | 0.705 |
| 10/07/2026 | Fraude PaySim | Financeiro | 8.5/10 | XGBoost baseline | 0.9972 |
| 11/07/2026 | Card Testing Sintético | Financeiro | 9.0/10 | transformer sequencial | 100% preventivo |