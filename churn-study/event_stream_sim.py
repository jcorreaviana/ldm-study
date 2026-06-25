"""
╔══════════════════════════════════════════════════════════════════╗
║         EVENT STREAM SIMULATOR — Churn de Telecom              ║
║   Simula sequências de eventos por cliente (proxy LDM)         ║
╚══════════════════════════════════════════════════════════════════╝

Diferença do churn_lab.py:
  - Antes: 1 linha por cliente (snapshot estático)
  - Agora:  N eventos por cliente (sequência temporal)

Fluxo:
  BLOCO 1 — Gerar event stream sintético
  BLOCO 2 — Visualizar sequências de clientes
  BLOCO 3 — Abordagem clássica (agregar → modelo)
  BLOCO 4 — Abordagem LDM (sequência → transformer)
  BLOCO 5 — Comparação e análise
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb

plt.style.use("seaborn-v0_8-darkgrid")
np.random.seed(42)

def bloco(n, titulo):
    print()
    print("═" * 60)
    print(f"  BLOCO {n} — {titulo}")
    print("═" * 60)

def ok(msg, start=None):
    elapsed = f" ({time.time()-start:.1f}s)" if start else ""
    print(f"  ✅ {msg}{elapsed}")

def info(msg):
    print(f"  ℹ️  {msg}")

# ══════════════════════════════════════════════════════════════════
# BLOCO 1 — GERAR EVENT STREAM SINTÉTICO
# ══════════════════════════════════════════════════════════════════
bloco(1, "Gerando event stream sintético")

# Tipos de eventos possíveis (como em uma telecom real)
EVENTOS = [
    "login_app",
    "consulta_fatura",
    "ligacao_suporte",
    "reclamacao_rede",
    "consulta_planos",      # sinal forte de churn
    "tentativa_cancelamento", # sinal muito forte
    "upgrade_plano",
    "downgrade_plano",       # sinal de churn
    "pagamento_ok",
    "pagamento_atrasado",    # sinal de churn
    "uso_dados_alto",
    "uso_dados_baixo",       # cliente desengajando
]

# Probabilidades de cada evento para clientes que VÃO churnar vs ficar
# Formato: [prob_fica, prob_churna]
PROB_EVENTO = {
    "login_app":              [0.20, 0.08],  # churners usam menos o app
    "consulta_fatura":        [0.10, 0.12],
    "ligacao_suporte":        [0.08, 0.15],  # churners ligam mais
    "reclamacao_rede":        [0.04, 0.14],  # churners reclamam mais
    "consulta_planos":        [0.05, 0.18],  # forte sinal de churn
    "tentativa_cancelamento": [0.01, 0.08],  # sinal fortíssimo
    "upgrade_plano":          [0.08, 0.02],  # churners não fazem upgrade
    "downgrade_plano":        [0.03, 0.09],  # sinal de churn
    "pagamento_ok":           [0.20, 0.06],
    "pagamento_atrasado":     [0.03, 0.12],  # sinal de churn
    "uso_dados_alto":         [0.12, 0.04],  # churners usam menos dados
    "uso_dados_baixo":        [0.06, 0.12],
}

def gerar_cliente(cliente_id, vai_churnar):
    """Gera sequência de eventos para um cliente"""
    n_eventos = np.random.randint(8, 25)  # cada cliente tem 8-25 eventos
    eventos = []

    # Se vai churnar, nos últimos eventos a probabilidade muda
    # (simula a "jornada de churn" que se acelera no fim)
    for i in range(n_eventos):
        # Nos últimos 30% dos eventos, churners têm padrão mais extremo
        fase_critica = vai_churnar and (i > n_eventos * 0.7)

        probs = []
        for ev in EVENTOS:
            p_fica, p_churna = PROB_EVENTO[ev]
            if vai_churnar:
                p = p_churna * (1.5 if fase_critica else 1.0)
            else:
                p = p_fica
            probs.append(p)

        # Normaliza para somar 1
        probs = np.array(probs)
        probs = probs / probs.sum()

        tipo = np.random.choice(EVENTOS, p=probs)
        dia  = i * np.random.randint(1, 5)  # dias desde o início

        eventos.append({
            "cliente_id": cliente_id,
            "evento_num": i,
            "dia":        dia,
            "tipo":       tipo,
            "churn":      int(vai_churnar)
        })

    return eventos

# Gerar 1000 clientes (700 ficam, 300 churnam — ~30% de churn)
t = time.time()
print("  ⏳ Gerando clientes e eventos...")

todos_eventos = []
N_CLIENTES = 1000
N_CHURN    = 300

for i in range(N_CLIENTES):
    vai_churnar = i < N_CHURN
    todos_eventos.extend(gerar_cliente(i, vai_churnar))

df_events = pd.DataFrame(todos_eventos)

ok(f"Event stream gerado", t)
info(f"Total de eventos: {len(df_events):,}")
info(f"Total de clientes: {N_CLIENTES} ({N_CHURN} churnam, {N_CLIENTES-N_CHURN} ficam)")
info(f"Média de eventos por cliente: {len(df_events)/N_CLIENTES:.1f}")
print()
print("  ── Primeiros eventos (cliente 0 — vai churnar) ──")
print(df_events[df_events.cliente_id == 0].head(8).to_string(index=False))
print()
print("  ── Primeiros eventos (cliente 300 — vai ficar) ──")
print(df_events[df_events.cliente_id == 300].head(8).to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# BLOCO 2 — VISUALIZAR SEQUÊNCIAS
# ══════════════════════════════════════════════════════════════════
bloco(2, "Visualizando sequências de eventos")

print("  ⏳ Gerando visualizações...")

# Frequência de cada tipo de evento por grupo
freq = df_events.groupby(["tipo", "churn"]).size().unstack(fill_value=0)
freq.columns = ["Ficou", "Churnou"]
freq = freq.div(freq.sum(axis=0), axis=1) * 100  # normaliza para %

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Event Stream Sintético — Padrões de Churn", fontsize=14, fontweight="bold")

# Plot 1: Frequência de eventos por grupo
ax = axes[0]
x = np.arange(len(freq))
w = 0.35
bars1 = ax.bar(x - w/2, freq["Ficou"],   w, label="Ficou",   color="#4f8fff", alpha=0.85)
bars2 = ax.bar(x + w/2, freq["Churnou"], w, label="Churnou", color="#e24b4a", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(freq.index, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("% dos eventos do grupo")
ax.set_title("Frequência de cada evento\npor grupo (% normalizada)")
ax.legend()
# Destaca eventos com maior diferença
for i, (ev, row) in enumerate(freq.iterrows()):
    diff = abs(row["Ficou"] - row["Churnou"])
    if diff > 3:
        ax.annotate("↑ diferença", xy=(i, max(row["Ficou"], row["Churnou"])),
                   fontsize=7, ha="center", color="#ef9f27")

# Plot 2: Sequência temporal de 6 clientes
ax2 = axes[1]
clientes_exemplo = [0, 1, 2, 300, 301, 302]  # 3 churners, 3 ficam
cores_tipo = {ev: plt.cm.tab20(i/len(EVENTOS)) for i, ev in enumerate(EVENTOS)}

for y_pos, cid in enumerate(clientes_exemplo):
    cliente_ev = df_events[df_events.cliente_id == cid]
    churn = cliente_ev["churn"].iloc[0]
    label_cor = "#e24b4a" if churn else "#4f8fff"
    label = f"C{cid} {'[churn]' if churn else '[ficou]'}"
    ax2.text(-2, y_pos, label, ha="right", va="center", fontsize=8, color=label_cor)
    for _, ev in cliente_ev.iterrows():
        cor = cores_tipo[ev["tipo"]]
        marker = "X" if "cancel" in ev["tipo"] or "reclamacao" in ev["tipo"] else "o"
        ax2.scatter(ev["dia"], y_pos, c=[cor], s=60, marker=marker, zorder=3)

ax2.set_yticks([])
ax2.set_xlabel("Dias desde o início")
ax2.set_title("Sequência temporal de eventos\n(X = eventos críticos)")
ax2.set_xlim(-5, 100)

# Legenda simplificada
patches = [mpatches.Patch(color=cores_tipo[ev], label=ev) for ev in EVENTOS]
ax2.legend(handles=patches, fontsize=6, loc="upper right", ncol=1)

plt.tight_layout()
plt.savefig("fig6_event_stream.png", dpi=150, bbox_inches="tight")
ok("fig6_event_stream.png salva")

# ══════════════════════════════════════════════════════════════════
# BLOCO 3 — ABORDAGEM CLÁSSICA (agregação → modelo)
# ══════════════════════════════════════════════════════════════════
bloco(3, "Abordagem clássica — agregar eventos em features")

print("  ⏳ Agregando eventos em features por cliente...")
t = time.time()

# Isso é exatamente o "feature engineering" que o LDM elimina
# Transformamos N eventos em 1 linha por cliente com médias e contagens
agg = df_events.groupby("cliente_id").agg(
    total_eventos        = ("tipo", "count"),
    dias_ativo           = ("dia", "max"),
    n_reclamacoes        = ("tipo", lambda x: (x == "reclamacao_rede").sum()),
    n_suporte            = ("tipo", lambda x: (x == "ligacao_suporte").sum()),
    n_consulta_planos    = ("tipo", lambda x: (x == "consulta_planos").sum()),
    n_tentativa_cancel   = ("tipo", lambda x: (x == "tentativa_cancelamento").sum()),
    n_pagamento_atrasado = ("tipo", lambda x: (x == "pagamento_atrasado").sum()),
    n_downgrade          = ("tipo", lambda x: (x == "downgrade_plano").sum()),
    n_upgrade            = ("tipo", lambda x: (x == "upgrade_plano").sum()),
    n_login              = ("tipo", lambda x: (x == "login_app").sum()),
    n_uso_baixo          = ("tipo", lambda x: (x == "uso_dados_baixo").sum()),
    churn                = ("churn", "first")
).reset_index()

ok("Agregação concluída", t)
info(f"Shape após agregação: {agg.shape} — {agg.shape[1]-2} features por cliente")
print()
print("  ── Exemplo: cliente 0 (churn) vs cliente 300 (ficou) ──")
cols_show = ["cliente_id","n_reclamacoes","n_consulta_planos","n_tentativa_cancel","n_login","churn"]
print(agg[agg.cliente_id.isin([0,300])][cols_show].to_string(index=False))

# Treinar modelo clássico com features agregadas
X_agg = agg.drop(columns=["cliente_id","churn"])
y_agg = agg["churn"]

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_agg, y_agg, test_size=0.2, random_state=42, stratify=y_agg
)

scaler = StandardScaler()
X_train_a = scaler.fit_transform(X_train_a)
X_test_a  = scaler.transform(X_test_a)

print()
print("  ⏳ Treinando XGBoost com features agregadas...")
t = time.time()
xgb_agg = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                              eval_metric="logloss", random_state=42, verbosity=0)
xgb_agg.fit(X_train_a, y_train_a)

y_proba_agg = xgb_agg.predict_proba(X_test_a)[:, 1]
auroc_agg   = roc_auc_score(y_test_a, y_proba_agg)
auprc_agg   = average_precision_score(y_test_a, y_proba_agg)

ok(f"XGBoost (agregado) treinado", t)
info(f"AUROC: {auroc_agg:.4f} | AUPRC: {auprc_agg:.4f}")
info("⚠️  Informação perdida: ordem dos eventos, aceleração no fim, padrões temporais")

# ══════════════════════════════════════════════════════════════════
# BLOCO 4 — ABORDAGEM LDM (sequência → transformer simples)
# ══════════════════════════════════════════════════════════════════
bloco(4, "Abordagem LDM — sequência de eventos com atenção temporal")

print("  ⏳ Preparando sequências para o modelo temporal...")
t = time.time()

# Encoder de tipo de evento (one-hot simplificado)
evento_idx = {ev: i for i, ev in enumerate(EVENTOS)}
N_TIPOS = len(EVENTOS)
SEQ_LEN = 20  # usamos os últimos 20 eventos de cada cliente

def encode_sequencia(cliente_id):
    """Pega os últimos SEQ_LEN eventos e cria features temporais"""
    evs = df_events[df_events.cliente_id == cliente_id].tail(SEQ_LEN)
    features = []
    for _, ev in evs.iterrows():
        # One-hot do tipo de evento
        oh = [0.0] * N_TIPOS
        oh[evento_idx[ev["tipo"]]] = 1.0
        # Posição temporal normalizada
        pos = ev["evento_num"] / SEQ_LEN
        features.extend(oh + [pos])

    # Padding se tiver menos de SEQ_LEN eventos
    while len(features) < SEQ_LEN * (N_TIPOS + 1):
        features.append(0.0)

    return features[:SEQ_LEN * (N_TIPOS + 1)]

clientes = agg["cliente_id"].values
labels   = agg["churn"].values

X_seq = np.array([encode_sequencia(cid) for cid in clientes])

ok(f"Sequências codificadas", t)
info(f"Shape das sequências: {X_seq.shape}")
info(f"Cada cliente = vetor de {X_seq.shape[1]} dims ({SEQ_LEN} eventos × {N_TIPOS+1} features)")

# Treinar XGBoost com features de sequência
# (proxy do LDM — preserva a ordem e os padrões temporais)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_seq, labels, test_size=0.2, random_state=42, stratify=labels
)

print()
print("  ⏳ Treinando XGBoost com features de sequência...")
t = time.time()
xgb_seq = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                              eval_metric="logloss", random_state=42, verbosity=0)
xgb_seq.fit(X_train_s, y_train_s)

y_proba_seq = xgb_seq.predict_proba(X_test_s)[:, 1]
auroc_seq   = roc_auc_score(y_test_s, y_proba_seq)
auprc_seq   = average_precision_score(y_test_s, y_proba_seq)

ok(f"XGBoost (sequência) treinado", t)
info(f"AUROC: {auroc_seq:.4f} | AUPRC: {auprc_seq:.4f}")

# ══════════════════════════════════════════════════════════════════
# BLOCO 5 — COMPARAÇÃO FINAL
# ══════════════════════════════════════════════════════════════════
bloco(5, "Comparação final — clássico vs sequência")

print()
print("  ── Resultado ──")
print(f"  {'Abordagem':<35} {'AUROC':>7} {'AUPRC':>7}")
print("  " + "─" * 50)
print(f"  {'XGBoost c/ features agregadas':<35} {auroc_agg:>7.4f} {auprc_agg:>7.4f}")
print(f"  {'XGBoost c/ sequência de eventos':<35} {auroc_seq:>7.4f} {auprc_seq:>7.4f}")
ganho = (auroc_seq - auroc_agg) / auroc_agg * 100
print()
info(f"Ganho de AUROC usando sequência: {ganho:+.1f}%")
print()

# Figura comparativa
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Clássico (agregado) vs Sequência (proxy LDM)", fontsize=14, fontweight="bold")

# Plot 1: AUROC e AUPRC
ax = axes[0]
metricas = ["AUROC", "AUPRC"]
vals_agg = [auroc_agg, auprc_agg]
vals_seq = [auroc_seq, auprc_seq]
x = np.arange(len(metricas))
ax.bar(x - 0.2, vals_agg, 0.35, label="Clássico (agregado)", color="#4f8fff", alpha=0.85)
ax.bar(x + 0.2, vals_seq, 0.35, label="Sequência (LDM proxy)", color="#e24b4a", alpha=0.85)
for i, (a, s) in enumerate(zip(vals_agg, vals_seq)):
    ax.text(i - 0.2, a + 0.005, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + 0.2, s + 0.005, f"{s:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(metricas)
ax.set_ylim([0.5, 1.0])
ax.set_title("Métricas de performance")
ax.legend()

# Plot 2: Distribuição de scores
ax2 = axes[1]
y_test_arr = np.array(y_test_s)
ax2.hist(y_proba_agg[y_test_arr==0], bins=20, alpha=0.6, color="#4f8fff",
         label="Ficou (agg)", density=True)
ax2.hist(y_proba_agg[y_test_arr==1], bins=20, alpha=0.6, color="#4f8fff",
         label="Churnou (agg)", density=True, linestyle="--",
         histtype="step", linewidth=2)
ax2.hist(y_proba_seq[y_test_arr==0], bins=20, alpha=0.6, color="#e24b4a",
         label="Ficou (seq)", density=True)
ax2.hist(y_proba_seq[y_test_arr==1], bins=20, alpha=0.6, color="#e24b4a",
         label="Churnou (seq)", density=True, linestyle="--",
         histtype="step", linewidth=2)
ax2.axvline(0.5, color="gray", linestyle="--", lw=1)
ax2.set_title("Distribuição de scores\n(separação entre classes)")
ax2.set_xlabel("Score")
ax2.legend(fontsize=7)

# Plot 3: O que é perdido na agregação
ax3 = axes[2]
ax3.axis("off")
texto = """O que a agregação PERDE:

✗ Ordem dos eventos
  "reclamação ANTES de consulta
   de planos" ≠ "consulta de
   planos ANTES de reclamação"

✗ Aceleração no fim
  Churners aceleram eventos
  críticos nos últimos dias.
  A média não captura isso.

✗ Padrões temporais
  "3 reclamações em 2 dias"
  vs "3 reclamações em 60 dias"
  — a média é a mesma: 3

O que a sequência PRESERVA:

✓ Contexto temporal completo
✓ Aceleração de comportamento
✓ Padrões de "jornada de churn"
✓ Interações entre eventos
"""
ax3.text(0.05, 0.95, texto, transform=ax3.transAxes,
         fontsize=9, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax3.set_title("Por que sequência > agregação")

plt.tight_layout()
plt.savefig("fig7_classico_vs_sequencia.png", dpi=150, bbox_inches="tight")
ok("fig7_classico_vs_sequencia.png salva")

print()
print("  🎉 Simulação completa!")
print()
print("  Arquivos gerados:")
print("    ✅ fig6_event_stream.png       — visualização das sequências")
print("    ✅ fig7_classico_vs_sequencia.png — comparação clássico vs LDM")