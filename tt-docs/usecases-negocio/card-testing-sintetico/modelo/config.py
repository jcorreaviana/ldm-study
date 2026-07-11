"""
Hiperparâmetros do transformer sequencial de card testing.
"""

from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass
class Config:
    # dados
    data_path: Path = HERE.parent / "dataset_sintetico.csv"
    out_dir: Path = HERE / "modelo_out"

    # janela de contexto
    # o padrão de card testing mínimo é micro -> micro -> golpe (3 eventos).
    # window_size=5 dá margem (até 4 transações anteriores + a atual).
    window_size: int = 5
    max_seq_len: int = 64  # trunca (mantém as mais recentes) clientes com sequências muito longas

    # arquitetura
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    cat_emb_dim: int = 8
    n_numeric: int = 4  # valor, hora, saldo_antes, saldo_depois

    # treino
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 5  # early stopping em épocas sem melhora do AUPRC de validação
    grad_clip: float = 1.0
    seed: int = 42

    # split de clientes (não de transações — a sequência de cada cliente
    # tem que ficar inteira em um único split, senão o modelo vaza contexto)
    val_size: float = 0.15
    test_size: float = 0.15

    # métrica de negócio: identificação de micro-transações precursoras do golpe
    micro_threshold: float = 10.0  # R$ — mesmo corte usado na EDA e no baseline
    n_precursores: int = 2  # quantas transações imediatamente antes do golpe checar
    alert_threshold: float = 0.5  # probabilidade mínima para considerar "alertado"
