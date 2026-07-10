# Resultado final — transformer sequencial (PaySim)

## Decisões de configuração confirmadas

Baseado no censo completo de `eda/analisar_namedest.py` (509.565 contas
`nameDest` únicas, TRANSFER/CASH_OUT):

| Decisão | Valor | Justificativa |
|---|---|---|
| Entidade da sequência | `nameDest` | `nameOrig` não se repete (ver EDA_paysim.md seção 5-7) |
| Label | por posição/transação | 5.062 contas (0,99%) são mistas (fraude + legítima) — label por conta classificaria errado as transações legítimas dessas contas |
| Janela de contexto | 13 eventos | p90 do censo = 13 (cobre 90% das contas sem truncar); mediana real é só 3 (curta demais); VRAM da RTX 5060 (~8,5GB) limita janelas maiores |
| Corte de vazamento | contexto = só eventos anteriores à posição rotulada | necessário para a meta "detectar antes da fraude principal" |
| Sinal de volume | fraco isoladamente | contas com fraude têm média 5,65 transações vs 5,43 nas legítimas — diferença desprezível; o transformer precisa aprender o padrão da sequência, não a contagem |
| Split treino/val/teste | por conta (`nameDest`), não por exemplo | `transformer_sequencial.py` usava `random_split()` sobre os exemplos, o que podia deixar transações da mesma conta em treino e validação ao mesmo tempo; corrigido para usar o campo `split` do `event_stream.npz` (gerado por `montar_event_stream.py` a partir do mesmo `split_por_conta()` de `preprocess_pipeline.py`) |

## Métricas (preenchidas por `avaliar_modelo.py`)

TEMPLATE — a tabela abaixo é sobrescrita com números reais depois que você
treinar o modelo e rodar a avaliação. Nenhuma métrica abaixo é real; não use
estes números para decisão nenhuma.

| Métrica | Valor |
|---|---|
| AUPRC (split de teste) | — |
| Recall | — |
| FPR (falso positivo) | — |
| TP / FP / FN / TN | — |

Meta técnica do script-desafio.md: AUPRC > 0.80 (comparar com baseline
tabular anterior de 0.705).

Para gerar de verdade:

```
python avaliar_modelo.py --data ../../paysim_data/event_stream.npz --checkpoint ../modelo/checkpoint_melhor_modelo.pt
```
