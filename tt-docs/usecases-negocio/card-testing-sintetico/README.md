# Script do Desafio — Card Testing Sequencial (Dataset Sintético)
> Terceiro projeto prático — primeiro onde o transformer sequencial é genuinamente necessário.
> Dataset: sintético gerado por gerar_dataset.py
> Diferencial: padrão card testing real onde 1 transação isolada NÃO detecta a fraude

---

## Por que dataset sintético?

```
PaySim:           padrão transacional (1 evento resolve com XGBoost)
                  transformer não era necessário

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
merchant     →  onde ocorreu
valor        →  amount
saldo_antes  →  oldbalance
saldo_depois →  newbalance
isFraud      →  0 ou 1 (só a transação grande é 1)
```

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
    micro 1: R$1-5 online  (isFraud=0)
    micro 2: R$1-5 online  (isFraud=0, +20-60 min)
    golpe:   R$500-2000    (isFraud=1, +30-90 min)
```

---

## Pré-requisitos

```bash
# dependências já instaladas do projeto anterior
py -3.12 -m pip install pandas numpy scikit-learn xgboost torch --index-url https://download.pytorch.org/whl/cu128
py -3.12 -m pip install matplotlib seaborn nvitop wandb
```

---

## Estrutura do Projeto

```
tt-docs/usecases-negocio/card-testing-sintetico/
│
├── dados/
│   └── gerar_dataset.py           →  gerador do dataset sintético
│
├── eda/
│   ├── eda_sintetico.py           →  análise exploratória
│   ├── EDA_sintetico.md           →  achados e decisões
│   └── eda_out/                   →  gráficos
│
├── preprocessing/
│   ├── montar_event_stream.py     →  sequências por clienteID
│   └── preprocess_pipeline.py     →  split + normalização
│
├── modelo/
│   ├── baseline_tabular.py        →  XGBoost (deve falhar aqui)
│   ├── transformer_sequencial.py  →  transformer (deve funcionar aqui)
│   └── checkpoint_melhor_modelo.pt
│
├── avaliacao/
│   ├── resultado_baseline.md      →  XGBoost no dataset sintético
│   ├── resultado_transformer.md   →  transformer no dataset sintético
│   ├── comparativo.md             →  baseline vs transformer
│   └── gpu_report.md              →  métricas da RTX 5060
│
├── gpu_monitoring/
│   ├── gpu_logger.py
│   └── wandb_config.py
│
├── troubleshooting.md             →  problemas e soluções
├── notas_desafio.md               →  decisões e aprendizados
└── README.md
```

---

## Passo a Passo

### FASE 1 — Gerar o Dataset

```bash
cd tt-docs/usecases-negocio/card-testing-sintetico/dados
py -3.12 gerar_dataset.py --n-legitimos 9000 --n-fraudadores 1000 --output dataset_sintetico.csv
```

Resultado esperado:
```
total de transações:  ~120.000-150.000
clientes únicos:      10.000
fraudes:              ~1.000 (0.7-1.0%)
ratio:                1:100 (mais balanceado que PaySim)
```

### FASE 2 — EDA

```
[ ]  distribuição de tipos de transação
[ ]  visualizar sequência de um cliente fraudador
     → confirmar padrão: micro → micro → golpe
[ ]  visualizar sequência de um cliente legítimo
     → confirmar ausência do padrão
[ ]  baseline simples: XGBoost sem contexto
     → esperado: AUPRC baixo (< 0.70)
     → confirma que 1 transação isolada não detecta
[ ]  decidir janela de contexto (mínimo 3 para capturar o padrão)
```

### FASE 3 — Pré-processamento

```
[ ]  montar event stream por clienteID (agora temos ID real!)
[ ]  z-score em valor, saldo_antes, saldo_depois
[ ]  encoding de tipo e merchant
[ ]  positional encoding temporal (delta entre transações)
[ ]  split por cliente (70/15/15)
[ ]  padding para sequências curtas
```

### FASE 4 — Baseline (deve falhar)

```
[ ]  XGBoost com features de 1 transação
[ ]  AUPRC esperado: < 0.70
[ ]  confirmar que padrão sequencial não é capturado
[ ]  documentar em resultado_baseline.md
```

### FASE 5 — Transformer Sequencial (deve funcionar)

```
[ ]  atenção entre transações do mesmo cliente
[ ]  janela mínima de 3 (captura micro1 → micro2 → golpe)
[ ]  treino na RTX 5060 com CUDA
[ ]  monitorar com nvitop + wandb
[ ]  AUPRC esperado: > 0.85
[ ]  documentar em resultado_transformer.md
```

### FASE 6 — Comparativo e Conclusão

```
[ ]  baseline vs transformer — gap de AUPRC
[ ]  provar que o transformer aprendeu o padrão sequencial
[ ]  visualizar: scores do modelo nas micro-transações
     → score deve subir após micro1 e micro2
     → confirma que o contexto está sendo usado
[ ]  relatório GPU
[ ]  notas do desafio + atualizar casos_de_uso.md
```

---

## Meta do Projeto

```
técnica:   AUPRC transformer > 0.85
           AUPRC baseline    < 0.70
           gap > 0.15        ← prova que transformer é necessário

negócio:   detectar card testing ANTES do golpe
           score sobe após micro-transações
           ação preventiva: segundo fator após 2 micro-transações seguidas

gpu:       treino < 30 minutos na RTX 5060
           temperatura < 83°C
           relatório wandb completo
```

---

## Diferenças vs Projetos Anteriores

| Aspecto | Credit Card | PaySim | Card Testing Sintético |
|---|---|---|---|
| clienteID | não | nameOrig (único) | sim — se repete |
| padrão | tabular | transacional | sequencial real |
| baseline | funciona bem | funciona muito bem | deve falhar |
| transformer | funciona | falha (sinal errado) | deve funcionar |
| dataset | público | público | criado por nós |
| controle do padrão | não | não | total |

---

## Lição que Este Desafio vai Provar

```
"quando o padrão é genuinamente sequencial e
 o contexto histórico é obrigatório para detectar,
 o transformer supera modelos tabulares simples"

isso é o argumento central do LDM da NeoSpace:
  dados de clientes têm padrões sequenciais ricos
  que modelos tabulares não conseguem capturar
  o LDM processa o event stream e aprende esses padrões
```