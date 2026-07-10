# Resultado do baseline tabular (XGBoost)

Contexto: o transformer sequencial (ancorado em `nameDest`) nao tem acesso a
`oldbalanceOrg`/`newbalanceOrig` (saldo de quem envia) - onde mora o sinal
classico de fraude do PaySim (TRANSFER que drena ~100% do saldo do
remetente, seguido de CASH_OUT imediato). Este baseline usa as features
tabulares completas (incluindo `erroBalanceOrig`) para checar se ha sinal
discriminativo alcancavel, antes de continuar ajustando o transformer.

## Dados

| | Treino | Validação | Teste |
|---|---|---|---|
| Linhas | 1,940,889 | 415,069 | 414,451 |
| Positivas (fraude) | 5,802 | 1,179 | 1,232 |

Split por `nameDest` (mesma função `split_por_conta`, seed=42, usada em
`preprocess_pipeline.py` e `montar_event_stream.py`) — nenhuma conta se
repete entre treino e validação.

## Resultado

| Métrica | Valor |
|---|---|
| Modelo | XGBoost |
| AUPRC (validação) | 0.9972 |
| Recall (threshold=0.5) | 0.9958 |
| TP / FP / FN / TN | 1174 / 6 / 5 / 413884 |
| Score médio — fraude | 0.9958 |
| Score médio — legítima | 0.0001 |
| Diferença de médias | +0.9957 |

Para comparação: o transformer sequencial tinha diferença de médias de
0.0001 (não discriminava nada). Se este baseline mostrar diferença
substancialmente maior, confirma que o problema do transformer é falta de
acesso às features do remetente (`oldbalanceOrg`/`newbalanceOrig`/
`erroBalanceOrig`), não arquitetura/otimização.

## Features mais importantes

| Feature | Importância |
|---|---|
| erroBalanceOrig | 0.6399 |
| newbalanceOrig | 0.2808 |
| erroBalanceDest | 0.0297 |
| amount | 0.0255 |
| oldbalanceOrg | 0.0072 |
| soma_valor_anterior | 0.0037 |
| soma_valor_janela_curta | 0.0035 |
| n_tx_anteriores | 0.0030 |
| max_valor_anterior | 0.0019 |
| newbalanceDest | 0.0016 |

## Comando para reproduzir

```
python baseline_tabular.py --path ../dataset.csv --modelo xgboost
```
