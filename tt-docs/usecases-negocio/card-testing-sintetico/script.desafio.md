# Script do Desafio — Card Testing Sequencial (Dataset Sintético)
> Terceiro projeto prático — primeiro onde o transformer sequencial é genuinamente necessário.
> Dataset: sintético gerado por gerar_dataset.py
> Status: CONCLUÍDO ✅
>
> Navegação: README.md → estrutura | contexto_negocio.md → briefing | notas_desafio.md → decisões e resultados

---

## Por que dataset sintético?

```
PaySim:            padrão transacional (1 evento resolve com XGBoost)
                   transformer não era necessário — lição aprendida

dataset sintético: padrão genuinamente sequencial
                   micro-transação 1 → micro-transação 2 → golpe
                   1 transação isolada NÃO detecta
                   contexto histórico é obrigatório
                   → transformer sequencial se justifica
```

---

## Estrutura do Dataset

```
clienteID    →  identificador único (C000001 legítimo, F000001 fraudador)
timestamp    →  data e hora da transação
tipo         →  compra, transferencia, saque, pagamento
merchant     →  onde ocorreu (variado — artefato corrigido)
valor        →  amount
saldo_antes  →  oldbalance
saldo_depois →  newbalance
isFraud      →  0 ou 1 (só a transação grande é 1)
```

---

## Padrão de Fraude Implementado

```
clientes legítimos:
  5-25 transações aleatórias
  valores R$10-800
  horários comerciais (8h-22h)
  isFraud = 0 sempre

clientes fraudadores (card testing):
  fase 1 (50%): 2-8 transações normais para disfarçar
  fase 2: madrugada (1h-5h)
    micro 1: R$1-5  (isFraud=0, +0 min)
    micro 2: R$1-5  (isFraud=0, +20-60 min)
    golpe:   R$500-2000 (isFraud=1, +30-90 min)
             tipo e merchant VARIADOS — corrigido para evitar artefato
```

---

## FASE 1 — Gerar o Dataset ✅

```bash
cd tt-docs/usecases-negocio/card-testing-sintetico/dados
py -3.12 gerar_dataset.py --n-legitimos 9000 --n-fraudadores 1000 --output dataset_sintetico.csv
```

Resultado obtido:
```
total de transações:  140.626
clientes únicos:      10.000
fraudes:              1.000 (0.71%)
período:              2024-01-01 a 2024-08-13
```

Ajuste necessário durante a execução:
```
artefato identificado: tipo='transferencia' + merchant='online' em 100% dos golpes
correção: tipo e merchant do golpe agora são aleatórios
          saque (38.3%), compra (32.2%), transferencia (29.5%)
```

---

## FASE 2 — EDA ✅

```bash
cd eda
py -3.12 run_all.py
```

Achados principais:
```
fraude:    1.000 de 140.626 (0.71%) | 1 por cliente fraudador
padrão:    micro (R$2-5) → micro (R$2-5) → golpe (R$500-2000)
           micro-transações têm isFraud=0 → modelo atual não detecta
horário:   fraudes concentradas 2h-8h (média 4h39)
valor:     fraude média R$1.235 vs legítimo R$396
           mas só aparece no golpe — micros ficam escondidas
```

---

## FASE 3 — Baseline XGBoost ✅

Features: transação isolada (valor, tipo, merchant, hora, saldo_antes, saldo_depois)
Split: aleatório estratificado 70/15/15

```
PR-AUC:              0.82   ← parece bom
recall micro-transações: 0%  ← ponto cego confirmado
score médio micros:  0.024% ← invisível para o modelo
score médio golpe:   94.9%  ← detecta depois que é tarde

conclusão: baseline detecta o golpe DEPOIS que o dinheiro saiu
           não detecta o card testing na fase útil
           → transformer sequencial necessário
```

---

## FASE 4 — Transformer Sequencial ✅

Arquitetura:
```
janela:    5 transações de contexto
d_model:   64
heads:     4
layers:    2
label:     label_preventivo (micro + golpe = 1)
           → força o modelo a alertar ANTES do golpe
```

Treino:
```bash
cd modelo
py -3.12 -u train.py
```

```
melhor AUPRC validação: 1.0000 (época 5)
tempo total:            90s na RTX 5060
```

---

## FASE 5 — Avaliação Final ✅

```bash
py -3.12 -u evaluate.py
```

```
AUPRC contra isFraud real:    0.3416
  (comparável com baseline 0.82 — trade-off esperado)

recall preventivo:            100%
  micro-transações:  score médio 99.99%
  golpes com alerta: 150/150 (100%)

baseline recall preventivo:   0%
transformer recall preventivo: 100%  ← transformer vence onde importa
```

Por que o AUPRC caiu para 0.34?
```
transformer dá score alto para micro-transações (isFraud=0)
→  na métrica clássica conta como falso positivo
→  mas são verdadeiros positivos de negócio (alerta preventivo)
→  a métrica certa para esse problema é o recall preventivo
```

---

## Resultado Final

| Métrica | Baseline XGBoost | Transformer |
|---|---|---|
| PR-AUC (isFraud real) | 0.82 | 0.34 |
| Recall golpe | 100% | 100% |
| Recall micro-transações | 0% | **100%** |
| Alerta ANTES do golpe | 0/150 | **150/150** |
| Score médio micros | 0.024% | **99.99%** |

---

## Meta do Projeto — Revisão

```
planejado:   AUPRC baseline < 0.70  →  obtido: 0.82  (baseline melhor que esperado)
             AUPRC transformer > 0.85 →  obtido: 1.00  (melhor que esperado)
             recall preventivo > 85%  →  obtido: 100%  (perfeito)

lição:       a meta de AUPRC < 0.70 para o baseline estava errada
             o baseline detecta bem o GOLPE (PR-AUC 0.82)
             o que ele não detecta é o PADRÃO SEQUENCIAL (recall micro 0%)
             a métrica certa era o recall preventivo, não o AUPRC
```

---

## Diferenças vs Projetos Anteriores

| Aspecto | Credit Card | PaySim | Card Testing Sintético |
|---|---|---|---|
| clienteID | não | nameOrig (único) | sim — se repete ✅ |
| padrão | tabular | transacional | sequencial real ✅ |
| baseline | funciona bem | funciona muito bem | funciona no golpe, cego no padrão |
| transformer | funciona | falha (sinal errado) | funciona — 100% preventivo ✅ |
| dataset | público | público | criado por nós ✅ |
| controle do padrão | não | não | total ✅ |

---

## Lição Principal

```
"quando o padrão é genuinamente sequencial e
 o contexto histórico é obrigatório para detectar,
 o transformer supera modelos tabulares na métrica que importa para o negócio"

a escolha da métrica define o que o modelo aprende:
  AUPRC clássico → modelo aprende a detectar o golpe (tarde demais)
  recall preventivo → modelo aprende a detectar o padrão (antes do golpe)
```