# app.py
# -*- coding: utf-8 -*-
"""
🦅 Dashboard Financeiro Preditivo — Harpia AeroDesign 2026
VERSÃO FINAL (v4): leitura multi-abas (Jan–Dez) + SARIMAX + XGBoost + Cenários + LGPD
"""
import os
import sys
import logging

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.etl import run_pipeline
from src.models import run_models
from src.scenarios import simulate_scenarios, compute_risk_metrics
from src.viz import (
    fig_historico_saldo,
    fig_previsao_ensemble,
    fig_cenarios_estocasticos,
    fig_comparacao_modelos,
)
from src.lgpd import hash_sha256

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(page_title="Harpia 2026 — Preditivo", page_icon="🦅", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PISO_LIQUIDEZ = 5000.0
URL_DADOS = (
    "https://raw.githubusercontent.com/yuriiih/HarpiaAerodesign_infraestrutura_dados/"
    "main/gastos%20harpia%202026.xlsx"
)
NOME_ARQUIVO_LOCAL = "gastos harpia 2026.xlsx"

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🦅 Harpia AeroDesign")
st.sidebar.markdown("### Dashboard Preditivo — 2026")
st.sidebar.markdown("---")
st.sidebar.markdown("**Piso de Liquidez:** R$ 5.000,00")
st.sidebar.markdown("**Horizonte:** 6 meses")
st.sidebar.markdown("**Modelos:** SARIMAX + XGBoost")
st.sidebar.markdown("---")
st.sidebar.caption("Yuri Pires · Projeto de Extensão III · UFABC")

# ============================================================
# CARGA + PIPELINE (cacheado)
# ============================================================
@st.cache_data(show_spinner="🔄 Carregando planilha (12 abas) e rodando Bronze → Silver → Gold...")
def carregar_e_processar():
    bronze = None
    try:  # 1º: URL raw do GitHub (Streamlit Cloud)
        bronze = pd.read_excel(URL_DADOS, sheet_name=None, header=None)
        logger.info("Bronze carregado da URL raw (todas as abas).")
    except Exception as e:
        logger.warning(f"URL raw falhou: {e}. Tentando arquivo local...")
    if bronze is None:  # 2º: arquivo local (Codespaces / dev)
        caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), NOME_ARQUIVO_LOCAL)
        if os.path.exists(caminho):
            bronze = pd.read_excel(caminho, sheet_name=None, header=None)
            logger.info("Bronze carregado do arquivo local (todas as abas).")
    if bronze is None:
        st.error("❌ Não foi possível carregar a planilha (URL e local falharam).")
        st.stop()
    return run_pipeline(bronze)


@st.cache_data(show_spinner="🧠 Treinando SARIMAX + XGBoost...")
def treinar_modelos(df_monthly: pd.DataFrame):
    return run_models({"monthly": df_monthly})


st.title("🦅 Dashboard Financeiro Preditivo — Harpia 2026")
st.markdown("*Diagnóstico, Previsão (ML) e Simulação de Cenários — SAE Brasil Aerodesign*")
st.markdown("---")

gold = carregar_e_processar()
df_monthly = gold["monthly"]
df_trans = gold["transactions"]
stats = gold["stats"]
saldo_atual = float(df_monthly["Saldo_Acumulado"].iloc[-1]) if len(df_monthly) else 0.0

# ============================================================
# ABAS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Diagnóstico", "🔮 Previsão (ML)", "🎲 Cenários", "🔒 LGPD", "🔍 Profiling (Raw)"]
)

# ---------------- TAB 1: DIAGNÓSTICO ----------------
with tab1:
    st.header("Diagnóstico Situacional")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saldo Atual", f"R$ {saldo_atual:,.2f}")
    c2.metric("Receita Média/mês", f"R$ {stats['mu_receita']:,.2f}")
    c3.metric("Despesa Média/mês", f"R$ {stats['mu_despesa']:,.2f}")
    cv = f"{(stats['sigma_despesa'] / stats['mu_despesa'] * 100):.1f}%" if stats["mu_despesa"] else "N/A"
    c4.metric("Coef. Variação Gastos", cv)

    if not df_monthly.empty:
        st.plotly_chart(fig_historico_saldo(df_monthly), use_container_width=True)

    with st.expander("📋 Camada Gold (mensal)"):
        st.dataframe(df_monthly.style.format("R$ {:.2f}"))
    if "checks" in gold and gold["checks"] is not None:
        with st.expander("🧾 Auditoria ETL (conferência com o Resumo de cada aba)"):
            st.dataframe(gold["checks"], use_container_width=True, hide_index=True)

# ---------------- TAB 2: PREVISÃO ML ----------------
with tab2:
    st.header("Previsão — Ensemble (SARIMAX + XGBoost)")
    if len(df_monthly) < 3:
        st.warning("⚠️ Histórico insuficiente para modelagem (mínimo 3 meses).")
    else:
        modelos = treinar_modelos(df_monthly)
        ens, sar, xgb = modelos["ensemble"], modelos["sarimax"], modelos["xgboost"]
        if ens.empty:
            st.warning("Modelos não treinados com estes dados.")
        else:
            st.plotly_chart(fig_previsao_ensemble(df_monthly, ens), use_container_width=True)
            with st.expander("🔬 Comparação de Modelos"):
                st.plotly_chart(fig_comparacao_modelos(sar, xgb, ens), use_container_width=True)
            pred = ens["Ensemble_Pred"]
            criticos = pred[pred < PISO_LIQUIDEZ]
            if not criticos.empty:
                st.error(f"🚨 **ALERTA:** projeção rompe o piso de R$ 5.000 em "
                         f"**{len(criticos)} mês(es):** {', '.join(criticos.index.strftime('%b/%Y'))}")
            else:
                st.success("✅ Projeção acima do piso em todo o horizonte.")

# ---------------- TAB 3: CENÁRIOS ----------------
with tab3:
    st.header("Simulação Estocástica de Cenários")
    from src.scenarios import _stats_robustos
    stats_robustos = _stats_robustos(gold)
    if stats_robustos.get('aporte_inicial_isolado', 0) > 0:
        st.info(f"ℹ️ Aporte inicial de R$ {stats_robustos['aporte_inicial_isolado']:,.2f} "
                f"isolado no cálculo de variabilidade (evento único, não recorrente).")
    st.markdown("Projeção por μ ± σ das receitas e despesas históricas.")
    scen = simulate_scenarios(stats_robustos, saldo_atual)
    risk = compute_risk_metrics(scen)
    st.plotly_chart(fig_cenarios_estocasticos(scen), use_container_width=True)

    st.subheader("Métricas de Risco")
    cols = st.columns(3)
    for i, (cen, emo) in enumerate([("Conservador", "🔴"), ("Moderado", "🟠"), ("Agressivo", "🟢")]):
        with cols[i]:
            m = risk[cen]
            st.markdown(f"### {emo} {cen}")
            st.metric("Saldo Mínimo Projetado", f"R$ {m['saldo_minimo']:,.2f}")
            st.write(f"**Mês crítico:** {m['mes_critico']}")
            if m["rompe_piso"]:
                st.error(f"⚠️ Rompe o piso em {6 - m['meses_acima_piso']} mês(es)")
            else:
                st.success("✅ Acima do piso")

# ---------------- TAB 4: LGPD ----------------
with tab4:
    st.header("🔒 Governança e LGPD")
    st.markdown("""
    - ✅ **SHA-256** — hash irreversível de identificadores de doadores
    - ✅ **k-Anonimato (k=5)** — mascaramento automático de grupos pequenos
    - ✅ **Sistema consultivo** — suporte à decisão, sem autonomia decisória
    """)
    st.subheader("Demonstração de Anonimização")
    demo = pd.DataFrame({
        "Doador": ["Doador#01", "Doador#02", "Rifa#104"],
        "Categoria": ["Aerosócio-Ouro", "Aerosócio-Prata", "Rifa Avulsa"],
        "Valor": [500.0, 300.0, 50.0],
    })
    demo["Hash_SHA256"] = demo["Doador"].apply(hash_sha256)
    st.dataframe(demo, use_container_width=True, hide_index=True)

# ---------------- TAB 5: PROFILING (LEGADO) ----------------
with tab5:
    st.header("🔍 Data Profiling — Qualidade dos Dados")
    nulos = int(df_trans["Valor"].isnull().sum())
    datas_inv = int(df_trans["Data"].isnull().sum())
    dups = int(df_trans.duplicated(subset=["Data", "Descricao", "Valor"]).sum())

    if nulos == 0:
        st.success("✅ Sucesso: Nenhum valor monetário nulo.")
    else:
        st.error(f"❌ Alerta: Detectados {nulos} registros sem valor monetário.")

    if datas_inv == 0:
        st.success(f"✅ Sucesso: Todas as {len(df_trans)} datas foram validadas corretamente.")
    else:
        st.error(f"❌ Erro: Inconsistência de tipo em {datas_inv} datas. Requer normalização ETL.")

    if dups == 0:
        st.success("✅ Sucesso: Nenhuma transação duplicada encontrada.")
    else:
        st.info(f"ℹ️ Aviso: Transações duplicadas identificadas: {dups}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Gastos", f"R$ {df_trans[df_trans.Tipo == 'Despesa']['Valor_Abs'].sum():,.2f}")
    c2.metric("Total de Receitas", f"R$ {df_trans[df_trans.Tipo == 'Receita']['Valor_Abs'].sum():,.2f}")
    c3.metric("Nº de Transações", f"{len(df_trans)}")
    c4.metric("Maior Saída Única", f"R$ {df_trans[df_trans.Tipo == 'Despesa']['Valor_Abs'].max():,.2f}")

    with st.expander("📋 Extrato Completo (Camada Silver)"):
        dfx = df_trans.copy()
        dfx["Data"] = dfx["Data"].dt.strftime("%d/%m/%Y")
        st.dataframe(dfx.sort_values("Data", ascending=False),
                     use_container_width=True, hide_index=True)