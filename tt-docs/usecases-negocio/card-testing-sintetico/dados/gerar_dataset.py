"""
GERADOR DE DATASET SINTÉTICO — Card Testing Sequencial
========================================================
Gera um dataset onde o padrão de fraude é genuinamente sequencial:
  micro-transação 1 (teste)  →  R$1-5
  micro-transação 2 (confirmação) →  R$1-5  (dentro de 1h)
  transação grande (golpe)   →  R$500-2000  (dentro de 2h)

Diferente do PaySim:
  - 1 transação isolada NÃO detecta a fraude
  - só o padrão histórico revela o card testing
  - demonstra genuinamente o valor do transformer sequencial

Uso:
  py -3.12 gerar_dataset.py --output dataset_sintetico.csv
  py -3.12 gerar_dataset.py --n-legitimos 9000 --n-fraudadores 1000 --output dataset_sintetico.csv
"""

import argparse
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# reproduzibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# PARÂMETROS DO DATASET
# ============================================================
TIPOS_TRANSACAO = ['compra', 'transferencia', 'saque', 'pagamento']
MERCHANTS = ['supermercado', 'farmacia', 'restaurante', 'posto', 'online', 'shopping']

DATA_BASE = datetime(2024, 1, 1)
DATA_FIM  = datetime(2024, 6, 30)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def timestamp_aleatorio(data_inicio=DATA_BASE, data_fim=DATA_FIM):
    """Gera timestamp aleatório dentro do período."""
    delta = data_fim - data_inicio
    segundos = random.randint(0, int(delta.total_seconds()))
    return data_inicio + timedelta(seconds=segundos)

def timestamp_horario_normal():
    """Timestamp em horário comercial (8h-22h) — padrão de cliente legítimo."""
    base = timestamp_aleatorio()
    hora = random.randint(8, 22)
    return base.replace(hour=hora, minute=random.randint(0, 59))

def timestamp_madrugada():
    """Timestamp em madrugada (1h-5h) — padrão suspeito."""
    base = timestamp_aleatorio()
    hora = random.randint(1, 5)
    return base.replace(hour=hora, minute=random.randint(0, 59))

def calcular_saldo_depois(saldo_antes, valor, tipo):
    """Calcula saldo após a transação."""
    if tipo in ['compra', 'pagamento', 'transferencia', 'saque']:
        return max(0, saldo_antes - valor)
    return saldo_antes + valor


# ============================================================
# GERADOR DE CLIENTE LEGÍTIMO
# ============================================================

def gerar_cliente_legitimo(clienteID):
    """
    Gera sequência de transações de um cliente legítimo.
    Padrão: compras aleatórias em horários normais, valores variados.
    Nenhuma transação é fraude.
    """
    n_transacoes = random.randint(5, 25)
    saldo = random.uniform(500, 10000)
    transacoes = []

    t = timestamp_horario_normal()

    for i in range(n_transacoes):
        # intervalo entre transações: horas a dias
        t += timedelta(hours=random.uniform(2, 72))
        valor = random.uniform(10, 800)
        tipo = random.choice(TIPOS_TRANSACAO)
        saldo_depois = calcular_saldo_depois(saldo, valor, tipo)

        transacoes.append({
            'clienteID':    clienteID,
            'timestamp':    t,
            'tipo':         tipo,
            'merchant':     random.choice(MERCHANTS),
            'valor':        round(valor, 2),
            'saldo_antes':  round(saldo, 2),
            'saldo_depois': round(saldo_depois, 2),
            'isFraud':      0
        })
        saldo = saldo_depois

    return transacoes


# ============================================================
# GERADOR DE CLIENTE FRAUDADOR (card testing)
# ============================================================

def gerar_cliente_fraude(clienteID):
    """
    Gera sequência de transações de um cliente fraudador.

    Padrão card testing:
      1. histórico normal (opcional) — para disfarçar
      2. micro-transação 1 — teste do cartão (R$1-5)
      3. micro-transação 2 — confirmação (R$1-5, dentro de 1h)
      4. transação grande — golpe (R$500-2000, dentro de 2h)
      5. sem mais transações — conta some

    Só a transação grande tem isFraud=1.
    As micro-transações são isFraud=0 — isso força o modelo
    a aprender o CONTEXTO para detectar a fraude.
    """
    transacoes = []
    saldo = random.uniform(200, 5000)

    # fase 1: histórico normal (50% dos fraudadores têm histórico)
    if random.random() > 0.5:
        n_normal = random.randint(2, 8)
        t = timestamp_horario_normal()

        for i in range(n_normal):
            t += timedelta(hours=random.uniform(12, 48))
            valor = random.uniform(20, 300)
            tipo = random.choice(TIPOS_TRANSACAO)
            saldo_depois = calcular_saldo_depois(saldo, valor, tipo)

            transacoes.append({
                'clienteID':    clienteID,
                'timestamp':    t,
                'tipo':         tipo,
                'merchant':     random.choice(MERCHANTS),
                'valor':        round(valor, 2),
                'saldo_antes':  round(saldo, 2),
                'saldo_depois': round(saldo_depois, 2),
                'isFraud':      0
            })
            saldo = saldo_depois

    # fase 2: card testing em madrugada
    t_golpe = timestamp_madrugada()

    # micro-transação 1 (teste)
    valor_micro1 = round(random.uniform(1, 5), 2)
    saldo_depois = calcular_saldo_depois(saldo, valor_micro1, 'compra')
    transacoes.append({
        'clienteID':    clienteID,
        'timestamp':    t_golpe,
        'tipo':         'compra',
        'merchant':     'online',
        'valor':        valor_micro1,
        'saldo_antes':  round(saldo, 2),
        'saldo_depois': round(saldo_depois, 2),
        'isFraud':      0   # ← micro-transação não é fraude isolada
    })
    saldo = saldo_depois

    # micro-transação 2 (confirmação, dentro de 30-60 min)
    t_golpe += timedelta(minutes=random.randint(20, 60))
    valor_micro2 = round(random.uniform(1, 5), 2)
    saldo_depois = calcular_saldo_depois(saldo, valor_micro2, 'compra')
    transacoes.append({
        'clienteID':    clienteID,
        'timestamp':    t_golpe,
        'tipo':         'compra',
        'merchant':     'online',
        'valor':        valor_micro2,
        'saldo_antes':  round(saldo, 2),
        'saldo_depois': round(saldo_depois, 2),
        'isFraud':      0   # ← micro-transação não é fraude isolada
    })
    saldo = saldo_depois

    # transação grande (golpe, dentro de 30-90 min da segunda micro)
    t_golpe += timedelta(minutes=random.randint(30, 90))
    valor_golpe = round(random.uniform(500, 2000), 2)
    # varia tipo e merchant do golpe para evitar artefato de simulação
    # modelo não pode aprender "transferencia+online = fraude" trivialmente
    tipo_golpe    = random.choice(['transferencia', 'saque', 'compra'])
    merchant_golpe = random.choice(MERCHANTS)
    saldo_depois = calcular_saldo_depois(saldo, valor_golpe, tipo_golpe)
    transacoes.append({
        'clienteID':    clienteID,
        'timestamp':    t_golpe,
        'tipo':         tipo_golpe,
        'merchant':     merchant_golpe,
        'valor':        valor_golpe,
        'saldo_antes':  round(saldo, 2),
        'saldo_depois': round(saldo_depois, 2),
        'isFraud':      1   # ← só aqui é fraude
    })

    return transacoes


# ============================================================
# GERADOR PRINCIPAL
# ============================================================

def gerar_dataset(n_legitimos=9000, n_fraudadores=1000):
    """
    Gera dataset completo com clientes legítimos e fraudadores.

    Args:
        n_legitimos:   número de clientes legítimos
        n_fraudadores: número de clientes fraudadores

    Returns:
        DataFrame ordenado por clienteID e timestamp
    """
    print(f"Gerando {n_legitimos} clientes legítimos...")
    todas_transacoes = []

    for i in range(n_legitimos):
        clienteID = f"C{i:06d}"
        todas_transacoes.extend(gerar_cliente_legitimo(clienteID))

    print(f"Gerando {n_fraudadores} clientes fraudadores...")
    for i in range(n_fraudadores):
        clienteID = f"F{i:06d}"
        todas_transacoes.extend(gerar_cliente_fraude(clienteID))

    df = pd.DataFrame(todas_transacoes)
    df = df.sort_values(['clienteID', 'timestamp']).reset_index(drop=True)

    return df


# ============================================================
# VALIDAÇÃO E ESTATÍSTICAS
# ============================================================

def validar_dataset(df):
    """Mostra estatísticas do dataset gerado."""
    print("\n" + "=" * 60)
    print("DATASET GERADO — ESTATÍSTICAS")
    print("=" * 60)
    print(f"  total de transações:  {len(df):,}")
    print(f"  clientes únicos:      {df['clienteID'].nunique():,}")
    print(f"  fraudes:              {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
    print(f"  período:              {df['timestamp'].min().date()} a {df['timestamp'].max().date()}")

    print(f"\n  transações por cliente:")
    stats = df.groupby('clienteID').size()
    print(f"    média:    {stats.mean():.1f}")
    print(f"    mediana:  {stats.median():.1f}")
    print(f"    mínimo:   {stats.min()}")
    print(f"    máximo:   {stats.max()}")

    print(f"\n  distribuição por tipo:")
    for tipo, count in df['tipo'].value_counts().items():
        print(f"    {tipo:15}  {count:,}")

    # validação do padrão card testing
    fraudadores = df[df['clienteID'].str.startswith('F')]['clienteID'].unique()
    print(f"\n  validação do padrão card testing:")
    print(f"    clientes fraudadores: {len(fraudadores):,}")

    # verifica se micro-transações precedem fraude
    exemplo = df[df['clienteID'] == fraudadores[0]].tail(5)
    print(f"\n  exemplo de sequência fraudulenta ({fraudadores[0]}):")
    print(exemplo[['timestamp', 'tipo', 'valor', 'isFraud']].to_string(index=False))

    print("\n" + "=" * 60)
    print("VERIFICAÇÃO: modelo simples deveria FALHAR aqui")
    print("=" * 60)
    print("""
  uma transação de R$2 em 'online' às 3h NÃO é fraude isolada
  só com o CONTEXTO das 2 micro-transações anteriores
  o modelo consegue prever que a próxima será o golpe

  → isso é o que força o transformer sequencial a ser necessário
  → XGBoost sem contexto não vai funcionar bem aqui
    (diferente do PaySim onde 1 feature resolvia tudo)
""")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset sintético de card testing")
    parser.add_argument("--n-legitimos",   type=int, default=9000,  help="número de clientes legítimos")
    parser.add_argument("--n-fraudadores", type=int, default=1000,  help="número de clientes fraudadores")
    parser.add_argument("--output",        type=str, default="dataset_sintetico.csv", help="arquivo de saída")
    args = parser.parse_args()

    df = gerar_dataset(args.n_legitimos, args.n_fraudadores)
    validar_dataset(df)

    df.to_csv(args.output, index=False)
    print(f"\n  dataset salvo em: {args.output}")
    print(f"  tamanho do arquivo: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")