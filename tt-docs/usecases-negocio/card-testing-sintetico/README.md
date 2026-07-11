# Card Testing Sintético — Detecção de Fraude Sequencial
> Terceiro projeto prático — primeiro onde o transformer sequencial é genuinamente necessário.
> Status: CONCLUÍDO ✅

---

## Navegação do Projeto

```
contexto_negocio.md          →  briefing do cliente (PayFlow)
                                use antes de começar para jogar o papel do Head de Riscos

script_desafio.md            →  plano original + o que foi executado
                                referência técnica de cada fase

notas_desafio.md             →  decisões tomadas, resultados e aprendizados
                                leitura principal após o projeto
```

---

## Estrutura de Arquivos

```
card-testing-sintetico/
│
├── dados/
│   └── gerar_dataset.py           →  gerador do dataset sintético
│                                     py -3.12 gerar_dataset.py --n-legitimos 9000 --n-fraudadores 1000
│
├── eda/
│   ├── utils.py                   →  funções compartilhadas
│   ├── 01_shape_dtypes.py         →  shape e tipos de coluna
│   ├── 02_distribuicao_fraude.py  →  desbalanceamento
│   ├── 03_distribuicao_tipos.py   →  tipos de transação
│   ├── 04_sequencia_fraudador.py  →  visualiza padrão micro → micro → golpe
│   ├── 05_sequencia_legitimo.py   →  confirma ausência do padrão
│   ├── 06_distribuicao_valores.py →  valores fraude vs legítimo
│   ├── 07_distribuicao_horarios.py→  horários fraude vs legítimo
│   ├── run_all.py                 →  roda todos os scripts de EDA
│   └── eda_out/                   →  gráficos gerados
│
├── baseline/
│   ├── train_baseline.py          →  XGBoost sem contexto sequencial
│   ├── evaluate_baseline.py       →  avaliação + ponto cego
│   └── metrics.json               →  PR-AUC 0.82, recall micro 0%
│
├── modelo/
│   ├── config.py                  →  hiperparâmetros (janela=5, d_model=64, 4 heads)
│   ├── utils.py                   →  carga, split, preprocessamento, label_preventivo
│   ├── dataset.py                 →  ClientSequenceDataset com máscara causal
│   ├── model.py                   →  TransformerEncoder + head de classificação
│   ├── train.py                   →  treino com BCE + early stopping por AUPRC
│   ├── evaluate.py                →  AUPRC real + recall preventivo
│   ├── run_all.py                 →  roda treino + avaliação
│   └── requirements.txt
│
├── modelo_out/
│   ├── checkpoint_melhor_modelo.pt →  melhor checkpoint (AUPRC val 1.0000)
│   ├── avaliacao_transformer.png   →  distribuição de scores por categoria
│   └── metrics.json                →  métricas completas
│
├── contexto_negocio.md            →  briefing do cliente
├── script_desafio.md              →  plano + execução
├── notas_desafio.md               →  decisões e resultados
└── README.md                      →  este arquivo
```

---

## Como Reproduzir

```bash
# 1. gerar o dataset
cd dados
py -3.12 gerar_dataset.py --n-legitimos 9000 --n-fraudadores 1000 --output dataset_sintetico.csv
cd ..

# 2. EDA
cd eda
py -3.12 run_all.py
cd ..

# 3. baseline (deve dar PR-AUC 0.82, recall micro 0%)
cd baseline
py -3.12 train_baseline.py
py -3.12 evaluate_baseline.py
cd ..

# 4. transformer (deve dar recall preventivo 100%)
cd modelo
py -3.12 -u train.py
py -3.12 -u evaluate.py
```

---

## Resultado em Uma Linha

```
baseline XGBoost:      PR-AUC 0.82  |  recall micro  0%  |  detecta o golpe depois
transformer sequencial: AUPRC 1.00   |  recall micro 100% |  detecta ANTES do golpe
```

---

## Lição Principal

```
a escolha da métrica define o que o modelo aprende
e o que o modelo aprende define se ele é útil para o negócio

AUPRC 0.82 que detecta após o prejuízo
< recall preventivo 100% que previne o golpe
```