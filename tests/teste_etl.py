# tests/teste_etl.py — v2 (iteração vetorial + σ recalculado da planilha)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src.etl import run_pipeline

ARQ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gastos harpia 2026.xlsx")
g = run_pipeline(ARQ)
gold, checks, stats = g["monthly"], g["checks"], g["stats"]

REC  = [10113.90, 4092.96, 5499.98, 4490.14, 1453.69, 1319.20, 3583.20, 40.00]
DESP = [3948.93, 0.00, 4847.80, 3774.55, 10245.63, 1996.82, 4195.04, 76.80]
SALD = [6164.97, 10257.93, 10910.11, 11625.70, 2833.76, 2156.14, 1544.30, 1507.50]

MU_ESP    = float(np.mean(DESP))
SIGMA_ESP = float(np.std(DESP, ddof=1))   # ≈ 3.254,90 (valor real da planilha)

falhas = []
def chk(nome, cond):
    print(("✅ " if cond else "❌ ") + nome)
    if not cond: falhas.append(nome)

# --- Gabarito mensal ---
chk(f"8 meses com movimentação (Jan–Ago): {len(gold)}", len(gold) == 8)
chk("Receitas mensais = Resumo das abas", np.allclose(gold["Receita"].values, REC, atol=0.01))
chk("Despesas mensais = Resumo das abas", np.allclose(gold["Despesa"].values, DESP, atol=0.01))
chk("Saldo acumulado = Tabela 1 do relatório", np.allclose(gold["Saldo_Acumulado"].values, SALD, atol=0.01))
chk(f"μ_gasto = {MU_ESP:,.2f}", abs(stats["mu_despesa"] - MU_ESP) < 0.01)
chk(f"σ_gasto (recalculado da planilha) = {SIGMA_ESP:,.2f}", abs(stats["sigma_despesa"] - SIGMA_ESP) < 0.5)
print(f"ℹ️  Errata: relatório publica σ = 3.412,87 (CV 93,9%); "
      f"planilha recalculada = {SIGMA_ESP:,.2f} (CV {SIGMA_ESP / MU_ESP * 100:.1f}%).")

# --- Conferências internas (vetorizadas, sem loop sobre DataFrame) ---
ck = checks.set_index("mes")
meses = gold.index.month
chk("Σ gasto diário = Σ saídas (todo mês)",
    float((ck.loc[meses, "gasto_diario"] - ck.loc[meses, "saidas"]).abs().max()) < 0.01)
chk("Resumo 'Faturamento' = Σ entradas (todo mês)",
    float((ck.loc[meses, "resumo_fat"].fillna(0) - ck.loc[meses, "entradas"]).abs().max()) < 0.01)
chk("Saldo final da aba = saldo acumulado Gold (todo mês)",
    float(np.abs(ck.loc[meses, "saldo_final"].values - gold["Saldo_Acumulado"].values).max()) < 0.01)
chk(f"Transações tipadas com descrição >= 150: {len(g['transactions'])}", len(g["transactions"]) >= 150)
chk("Sem valores zero e sem datas NaT no Silver",
    bool((g["transactions"]["Valor_Abs"] > 0).all() and g["transactions"]["Data"].notna().all()))

print("\n" + ("🎉 TODOS OS TESTES PASSARAM" if not falhas else f"⚠️ {len(falhas)} FALHA(S): {falhas}"))
sys.exit(1 if falhas else 0)