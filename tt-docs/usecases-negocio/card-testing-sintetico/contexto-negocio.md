# Contexto de Negócio — Card Testing Sintético
> Briefing do cliente para simulação do caso de uso.
> Use esse script para jogar o papel do Head de Riscos na reunião inicial.

---

## O Cliente

**Empresa:** PayFlow — fintech brasileira de pagamentos digitais
**Segmento:** cartões pré-pagos para população desbancarizada
**Base:** 2 milhões de cartões ativos
**Ticket médio:** R$85 por transação

---

## O Problema

*"Nos últimos 4 meses, estamos vendo um crescimento de 340% em chargebacks por card testing. O padrão é sempre o mesmo: alguém faz duas ou três comprinhas de R$1 a R$5 em sites online, e menos de 2 horas depois faz uma transferência de R$500 a R$2.000 que a gente acaba revertendo.*

*Nosso modelo atual analisa cada transação de forma isolada. Ele bloqueia transações acima de R$1.000 em horário de madrugada, mas os fraudadores aprenderam a ficar abaixo desse limite. E aí a gente tem dois problemas: deixa as fraudes passarem e ainda bloqueia clientes legítimos que fazem compras grandes à noite.*

*Trouxe aqui um dataset com 140 mil transações dos últimos 6 meses. Quero entender se dá para construir algo que enxergue esse padrão antes de a fraude acontecer."*

---

## Números do Problema

```
base de cartões:          2.000.000
chargebacks por mês:      ~R$1.2M  (cresceu 340% em 4 meses)
ticket médio do golpe:    R$850
fraudes por mês:          ~1.400 casos
falsos positivos atuais:  8% das transações legítimas bloqueadas
```

---

## Modelo Atual

```
regra 1:  bloqueia transações > R$1.000 entre 23h e 6h
regra 2:  bloqueia se destino é conta nova (< 30 dias)
regra 3:  bloqueia se 3+ transações em 1 hora

problema com regra 1:  fraudadores ficam abaixo de R$1.000
problema com regra 2:  não captura contas antigas usadas como mulas
problema com regra 3:  bloqueia clientes legítimos que pagam várias contas
```

---

## O que o Cliente Quer

```
detectar o padrão:
  micro-transação 1 (R$1-5)  →  sinal de alerta
  micro-transação 2 (R$1-5)  →  confirmação do padrão
  transação grande (R$500+)  →  bloquear ANTES de acontecer

ação desejada:
  após 2 micro-transações em menos de 1h
  pedir segundo fator de autenticação na próxima transação
  independente do valor

critérios de sucesso:
  recall:     capturar > 85% dos casos de card testing
  FPR:        bloquear < 3% das transações legítimas
  latência:   decisão em < 200ms (tempo real)
```

---

## Perguntas que o Cliente Vai Fazer

```
1. "vocês conseguem detectar antes do golpe ou só depois?"
   → resposta esperada: detectamos após as micro-transações,
     antes da transação grande

2. "e se o fraudador espaçar mais as micro-transações?"
   → resposta esperada: a janela de contexto cobre X horas,
     podemos ajustar conforme o padrão evolui

3. "como eu explico para o cliente legítimo que foi bloqueado?"
   → resposta esperada: segundo fator, não bloqueio;
     o cliente confirma e segue normalmente

4. "quanto tempo para colocar em produção?"
   → resposta esperada: shadow mode 30 dias,
     depois gradual rollout começando com 10% da base
```

---

## Diferença para os Projetos Anteriores

```
credit card (Kaggle):
  sem clienteID → transformer tabular
  features anônimas (PCA) → não interpretável
  padrão: features isoladas discriminam

PaySim:
  nameOrig único → sem event stream real
  padrão transacional → XGBoost resolve
  lição: baseline primeiro!

card testing sintético:
  clienteID real que se repete → event stream genuíno
  padrão sequencial → 1 transação não detecta
  lição: aqui o transformer é obrigatório
```