Esta pasta é onde train.py e evaluate.py salvam seus artefatos:

  - preprocessors.pkl        (gerado por train.py)
  - model.pt                 (gerado por train.py -- melhor checkpoint por AUPRC de validação)
  - model_config.json        (gerado por train.py)
  - train_history.json       (gerado por train.py)
  - test_client_ids.json     (gerado por train.py)
  - avaliacao_transformer.png (gerado por evaluate.py)
  - metrics.json             (gerado por evaluate.py)

O transformer sequencial ainda não foi treinado nesta sessão (todo o
desenvolvimento em modelo/ foi feito sob a instrução explícita de só
criar/corrigir os scripts, sem executar). Por isso esta pasta está vazia
por enquanto -- os arquivos acima aparecem aqui automaticamente na primeira
vez que você rodar:

    pip install -r ../modelo/requirements.txt
    cd modelo
    python train.py
    python evaluate.py

(ou `python run_all.py`, que roda os dois em sequência)
