# modelo/

`checkpoint_melhor_modelo.pt` não está incluído neste repositório — é gerado
por `transformer_sequencial.py` quando você roda o treino de verdade na sua
RTX local. Não fabriquei um checkpoint falso aqui porque um arquivo `.pt`
com pesos inventados seria enganoso (e inútil — não carregaria um modelo
que realmente aprendeu nada).

Para gerar:

```
python ../preprocessing/montar_event_stream.py --path ../../dataset.csv --output-dir ../../paysim_data
python transformer_sequencial.py --data ../../paysim_data/event_stream.npz
```

O checkpoint aparece aqui automaticamente ao final do treino (salvo a cada
época em que o AUPRC de validação melhora).
