# Notas do Desafio — PaySim Mobile Money Fraud Detection
> Dataset: PaySim (Kaggle — ealaxi/paysim1)
> Formato: par híbrido — José (negócio + validação) + Claude/Cowork (implementação)
> Status: ENCERRADO — limitação do dataset identificada, migração para dataset sintético

---

## Estrutura do Projeto

```
tt-docs/usecases-negocio/paysim-fraud/
│
├── eda/
│   ├── EDA_paysim.md              →  análise exploratória completa
│   ├── analisar_namedest.py       →  censo de sequências por nameDest
│   └── eda_out/                   →  gráficos gerados
│
├── preprocessing/
│   ├── montar_event_stream.py     →  monta sequências por nameDest + z-score
│   ├── preprocess_pipeline.py     →  features tabulares + split por conta
│   └── preprocessing_meta.json   →  estatísticas de normalização
│
├── modelo/
│   ├── transformer_sequencial.py  →  transformer com atenção entre transações
│   └── baseline_tabular.py        →  XGBoost com features do remetente
│
├── avaliacao/
│   ├── resultado_baseline.md      →  AUPRC 0.9972 (XGBoost)
│   └── resultado_final.md         →  comparativo transformer vs baseline
│
├── troubleshooting_paysim.md      →  8 problemas documentados
└── README.md
```

---

## Contexto de Negócio

**Cliente simulado:** operadora de mobile money com fraudes crescendo 180% em 3 meses

**Problema:** modelo atual bloqueia transações acima de R$10.000. Fraudadores fazem transferências menores para driblar o limite (card testing).

**Dataset:** PaySim — 6.3 milhões de transações sintéticas de mobile money africano.

---

## Caso de Uso — Score: 8.5/10

**Data:** 10/07/2026

### Pontos fortes
```
✓  quantificou impacto financeiro (180% de crescimento = modelo inviável)
✓  identificou card testing como padrão sequencial
✓  justificou clienteID como diferencial da arquitetura
✓  conectou event stream com LDM de forma natural
✓  shadow mode e gradual rollout mencionados organicamente
✓  label por posição escolhido corretamente (tempo real vs estático)
```

### Oportunidades de melhoria
```
→  acurácia não é métrica adequada para dataset desbalanceado
→  demorou para nomear "sequencial" como tipo de problema
→  não mencionou tipo específico de fraude mais custoso logo no início
```

---

## EDA — Achados Principais

```
6.362.620 transações | 8.213 fraudes | 0.129% desbalanceamento (1:774)

tipos com fraude:
  CASH_OUT:  50.1%  ← únicas com fraude
  TRANSFER:  49.9%  ← únicas com fraude
  PAYMENT, CASH_IN, DEBIT: ZERO fraudes

padrão de fraude:
  TRANSFER que drena 100% do saldo do remetente
  seguido por CASH_OUT imediato
  → regra de 1 transação, não padrão sequencial
```

---

## Decisões Arquiteturais e Suas Justificativas

### nameOrig vs nameDest como âncora

```
problema descoberto: nameOrig aparece UMA única vez por cliente
                     não há histórico sequencial pelo remetente

solução: pivotou para nameDest
  contas de destino se repetem (até 75 vezes)
  narrativa: detectar contas laranja/mulas
  label por posição: cada transação recebida avaliada com histórico anterior
```

### Janela de contexto = 13

```
distribuição de sequências por nameDest:
  mediana:       3 transações
  percentil 90:  13 transações
  percentil 99:  28 transações

decisão: janela=13 cobre 90% das contas
  sem truncar, com padding para o resto
  adequado para VRAM de 8.5GB (RTX 5060)
```

### Split por conta (não por exemplo)

```
problema detectado: random_split() por exemplo causa data leakage
  mesma conta em treino E validação
  AUPRC inflado artificialmente

solução: split por conta
  campo 'split' salvo no event_stream.npz
  mesma conta nunca aparece em treino e validação
```

---

## Troubleshooting — Resumo dos 8 Problemas

| # | Problema | Causa | Solução |
|---|---|---|---|
| 1 | PyTorch não instalava | Python 3.14 incompatível | Instalar Python 3.12 |
| 2 | GPU não reconhecida | RTX 5060 Blackwell precisa CUDA 12.8 | PyTorch 2.11 + cu128 |
| 3 | perda=nan época 1 | autocast deprecated + loss em fp16 | Mover loss para fora do autocast |
| 4 | pos_weight overflow | ratio 1:836 → pos_weight=836 explode | Capar em 50 + FocalLoss |
| 5 | perda=nan persistindo | features sem z-score (valores em milhões) | z-score em amount/balance |
| 6 | data leakage no split | random_split() por exemplo | Split por conta no .npz |
| 7 | GPU com 12% utilização | DataLoader sem num_workers | num_workers=4 pin_memory=True |
| 8 | AUPRC 0.0016 estagnado | features do remetente ausentes | Identificado via baseline |

---

## Resultado Final

### Transformer Sequencial
```
AUPRC validação:  0.0016  ← aleatório
scores fraude:    média 0.0487
scores legítimo:  média 0.0486
diferença:        0.0001  ← modelo não discrimina
causa:            features do remetente ausentes (92% do sinal)
```

### Baseline XGBoost (resultado real)
```
AUPRC validação:  0.9972  ← quase perfeito
scores fraude:    média 0.9958
scores legítimo:  média 0.0001
diferença:        0.9957  ← separação perfeita

top features:
  erroBalanceOrig:  63.9%
  newbalanceOrig:   28.1%
  erroBalanceDest:   3.0%
  amount:            2.6%
```

---

## Limitação Fundamental do PaySim

```
o padrão de fraude do PaySim é uma regra de 1 transação:
  erroBalanceOrig ≈ 1  →  remetente drena 100% do saldo
  → fraude detectada com 99.72% de certeza por uma regra simples

não é um padrão sequencial:
  não requer histórico do cliente
  não requer atenção entre transações
  um XGBoost com as features certas resolve trivialmente

conclusão: PaySim não é adequado para demonstrar LDM/transformer sequencial
```

---

## Lições Aprendidas

### Técnicas
```
1. SEMPRE rodar baseline antes do transformer
   se AUPRC > 0.95 com modelo simples → problema não precisa de transformer

2. features do remetente vs destinatário importam
   a entidade que ancora o event stream determina quais features estão disponíveis
   pivotando de nameOrig para nameDest perdemos 92% do sinal

3. z-score é obrigatório antes de nn.Linear
   valores na casa de milhões → overflow → perda=nan
   normalizar ANTES de salvar no .npz, não dentro do modelo

4. split por entidade, não por exemplo
   em dados sequenciais: mesma entidade nunca em treino E validação

5. DataLoader é gargalo comum
   num_workers=4 + pin_memory=True → GPU de 12% para 19%+

6. autocast + loss: loss FORA do autocast
   dentro do autocast = fp16 → overflow com pos_weight alto
```

### De Negócio
```
1. o tipo de padrão determina a arquitetura
   padrão transacional (1 evento) → modelo tabular
   padrão sequencial (contexto histórico) → transformer/LDM

2. baseline honesto é mais valioso que transformer complexo
   AUPRC 0.9972 com XGBoost > AUPRC 0.0016 com transformer
   escolher o modelo certo para o problema, não o mais sofisticado

3. limitação do dataset sintético
   PaySim simula fraude simples (drenagem de saldo)
   dados reais têm padrões mais complexos e sequenciais
   dataset sintético criado especificamente para o problema é mais adequado
```

---

## Próximo Desafio — Dataset Sintético

**Objetivo:** criar dataset com padrão genuinamente sequencial

**Características necessárias:**
```
clienteID que se repete múltiplas vezes
padrão card testing real:
  micro-transação 1 (R$1-10) → teste do cartão
  micro-transação 2 (R$1-10) → confirmação
  transação grande (R$500+)  → golpe real
label que exige sequência:
  1 transação isolada NÃO detecta
  só o padrão histórico revela a fraude
comportamento temporal claro:
  micro-transações ocorrem em janela de 24h antes do golpe
```

**Vantagem:** controlamos o padrão → sabemos que o transformer vai precisar da sequência para detectar.

---

## Comparativo de Desafios

| Data | Caso | Domínio | Score | AUPRC | Modelo |
|---|---|---|---|---|---|
| ~05/07/2026 | Inadimplência Fintech | Financeiro | 7.5/10 | — | conceitual |
| ~06/07/2026 | Churn Operadora | Telecom | 8.0/10 | — | conceitual |
| 07/07/2026 | Recomendação Produtos | Varejo | 8.5/10 | — | conceitual |
| 10/07/2026 | Fraude Credit Card | Financeiro | 8.5/10 | 0.705 | transformer tabular |
| 10/07/2026 | Fraude PaySim | Financeiro | 8.5/10 | 0.9972 | XGBoost (baseline) |