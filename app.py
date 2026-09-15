# app.py
# -*- coding: utf-8 -*-
"""
🦅 Dashboard Financeiro Preditivo — Harpia AeroDesign 2026
Versão final unificada: leitura via URL raw (GitHub) + pipeline preditivo completo
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging

# Adiciona src ao path (permite importar módulos do pacote src/)
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

# ============================================================
# 1. CONFIGURAÇÃO INICIAL
# ============================================================
st.set_page_config(
    page_title="Harpia 2026 — Preditivo",
    page_icon="🦅",
    layout="wide",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 2. SIDEBAR
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
# 3. CARREGAMENTO DE DADOS (VERSÃO ROBUSTA — SUA LÓGICA + EXTRAS)
# ============================================================
# URL RAW do seu repositório GitHub (mantida do seu código)
URL_DADOS = (
    "https://raw.githubusercontent.com/yuriiih/HarpiaAerodesign_infraestrutura_dados/"
    "main/gastos%20harpia%202026.xlsx"
)
NOME_ARQUIVO_LOCAL = "gastos harpia 2026.xlsx"


@st.cache_data(show_spinner="🔄 Carregando planilha de gastos...")
def carregar_dados_brutos():
    """
    Carrega dados da planilha com múltiplos fallbacks:
    1. URL raw do GitHub (Streamlit Cloud)
    2. Arquivo local (desenvolvimento)
    """
    df_raw = None

    # 1º tentativa: URL raw do GitHub (sua lógica)
    try:
        logger.info("Tentando carregar via URL raw do GitHub...")
        df_raw = pd.read_excel(URL_DADOS, header=None)
        logger.info("✅ Sucesso: dados carregados da URL raw.")
    except Exception as e:
        logger.warning(f"URL raw falhou: {e}. Tentando arquivo local...")

    # 2º tentativa: arquivo local no mesmo diretório do script
    if df_raw is None:
        diretorio_script = os.path.dirname(os.path.abspath(__file__))
        caminho_local = os.path.join(diretorio_script, NOME_ARQUIVO_LOCAL)
        if os.path.exists(caminho_local):
            try:
                df_raw = pd.read_excel(caminho_local, header=None)
                logger.info("✅ Sucesso: dados carregados do arquivo local.")
            except Exception as e:
                logger.error(f"Erro ao ler arquivo local: {e}")

    if df_raw is None:
        st.error(
            "❌ **ERRO CRÍTICO:** não foi possível carregar a planilha.\n\n"
            f"• URL tentada: `{URL_DADOS}`\n"
            f"• Arquivo local: `{NOME_ARQUIVO_LOCAL}`\n\n"
            "Verifique se o arquivo foi commitado na branch `main` com o nome exato "
            "(Linux diferencia maiúsculas de minúsculas)."
        )
        st.stop()

    return df_raw


@st.cache_data(show_spinner="🔄 Processando pipeline Bronze → Silver → Gold...")
def executar_pipeline_completo(df_raw):
    """Executa o pipeline ETL Medallion completo."""
    return run_pipeline(df_raw)


# ============================================================
# 4. EXECUÇÃO
# ============================================================
st.title("🦅 Dashboard Financeiro Preditivo — Harpia 2026")
st.markdown(
    "*Diagnóstico, Previsão (ML) e Simulação de Cenários — SAE Brasil Aerodesign*"
)
st.markdown("---")

try:
    df_raw = carregar_dados_brutos()
    gold_data = executar_pipeline_completo(df_raw)
except Exception as e:
    st.error(f"❌ Erro no pipeline: {e}")
    logger.exception("Detalhes do erro:")
    st.stop()

df_monthly = gold_data["monthly"]
df_trans = gold_data["transactions"]
stats = gold_data["stats"]
saldo_atual = (
    df_monthly["Saldo_Acumulado"].iloc[-1]
    if len(df_monthly) > 0
    else 0.0
)

# ============================================================
# 5. ABAS DO DASHBOARD
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Diagnóstico",
        "🔮 Previsão (ML)",
        "🎲 Cenários Estocásticos",
        "🔒 Governança LGPD",
        "🔍 Profiling (Raw)",
    ]
)

# ==================== TAB 1: DIAGNÓSTICO ====================
with tab1:
    st.header("Diagnóstico Situacional")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saldo Atual", f"R$ {saldo_atual:,.2f}")
    c2.metric(
        "Receita Média/mês",
        f"R$ {stats['mu_receita']:,.2f}",
    )
    c3.metric(
        "Despesa Média/mês",
        f"R$ {stats['mu_despesa']:,.2f}",
    )
    cv_gastos = (
        f"{(stats['sigma_despesa'] / stats['mu_despesa'] * 100):.1f}%"
        if stats["mu_despesa"] > 0
        else "N/A"
    )
    c4.metric("Coef. Variação Gastos", cv_gastos)

    if not df_monthly.empty:
        st.plotly_chart(
            fig_historico_saldo(df_monthly), use_container_width=True
        )

    with st.expander("📋 Dados Mensais Agregados (Camada Gold)"):
        st.dataframe(df_monthly.style.format("R$ {:.2f}"))

# ==================== TAB 2: PREVISÃO ML ====================
with tab2:
    st.header("Previsão — Ensemble (SARIMAX + XGBoost)")

    if len(df_monthly) < 3:
        st.warning(
            "⚠️ Dados insuficientes para modelos preditivos robustos. "
            "Mínimo recomendado: 3 meses de histórico."
        )
        st.info("💡 Use a aba **Cenários** para projeções estatísticas.")
    else:
        with st.spinner("Treinando modelos..."):
            modelos = run_models(gold_data)

        ensemble = modelos["ensemble"]
        sarimax = modelos["sarimax"]
        xgb = modelos["xgboost"]

        if ensemble.empty:
            st.warning("Modelos não puderam ser treinados.")
        else:
            st.plotly_chart(
                fig_previsao_ensemble(df_monthly, ensemble),
                use_container_width=True,
            )

            with st.expander("🔬 Comparação de Modelos"):
                st.plotly_chart(
                    fig_comparacao_modelos(sarimax, xgb, ensemble),
                    use_container_width=True,
                )

            # Alertas de ruptura
            if "Ensemble_Pred" in ensemble.columns:
                pred = ensemble["Ensemble_Pred"]
                criticos = pred[pred < 5000]
                if not criticos.empty:
                    meses_txt = ", ".join(criticos.index.strftime("%b/%Y"))
                    st.error(
                        f"🚨 **ALERTA:** projeção rompe piso de R$ 5.000 em "
                        f"**{len(criticos)} mês(es):** {meses_txt}"
                    )
                else:
                    st.success(
                        "✅ Projeção mantém saldo acima do piso em todo o horizonte."
                    )

# ==================== TAB 3: CENÁRIOS ESTOCÁSTICOS ====================
with tab3:
    st.header("Simulação Estocástica de Cenários")
    st.markdown("Baseado em μ ± σ das receitas e despesas históricas.")

    scenarios_df = simulate_scenarios(stats, saldo_atual)
    risk = compute_risk_metrics(scenarios_df)

    st.plotly_chart(
        fig_cenarios_estocasticos(scenarios_df), use_container_width=True
    )

    # Métricas de risco
    st.subheader("Métricas de Risco")
    cols = st.columns(3)
    for i, (cenario, emoji) in enumerate(
        [("Conservador", "🔴"), ("Moderado", "🟠"), ("Agressivo", "🟢")]
    ):
        with cols[i]:
            m = risk[cenario]
            st.markdown(f"### {emoji} {cenario}")
            st.metric("Saldo Mínimo Projetado", f"R$ {m['saldo_minimo']:,.2f}")
            st.write(f"**Mês Crítico:** {m['mes_critico']}")
            if m["rompe_piso"]:
                st.error(f"⚠️ Rompe piso em {6 - m['meses_acima_piso']} mês(es)")
            else:
                st.success("✅ Acima do piso")

    # Simulador interativo
    st.markdown("---")
    st.subheader("🎛️ Simulador Interativo (What-if)")
    col_a, col_b = st.columns(2)
    with col_a:
        rec_mult = st.slider("Multiplicador de Receita", 0.5, 2.0, 1.0, 0.1)
    with col_b:
        desp_mult = st.slider("Multiplicador de Despesa", 0.5, 2.0, 1.0, 0.1)

    stats_custom = stats.copy()
    stats_custom["mu_receita"] = stats["mu_receita"] * rec_mult
    stats_custom["mu_despesa"] = stats["mu_despesa"] * desp_mult

    custom_scenarios = simulate_scenarios(stats_custom, saldo_atual)
    st.plotly_chart(
        fig_cenarios_estocasticos(custom_scenarios), use_container_width=True
    )

# ==================== TAB 4: LGPD ====================
with tab4:
    st.header("🔒 Governança e LGPD")
    from src.lgpd import hash_sha256, aplicar_k_anonimato

    st.markdown(
        """
        **Protocolos implementados:**
        - ✅ **SHA-256** — Hash irreversível de identificadores de doadores
        - ✅ **k-Anonimato (k=5)** — Mascaramento automático de grupos pequenos
        - ✅ **Sistema consultivo** — Suporte à decisão, sem autonomia decisória
        """
    )

    st.subheader("Demonstração de Anonimização")
    demo = pd.DataFrame(
        {
            "Doador": ["Doador#01", "Doador#02", "Rifa#104"],
            "Categoria": ["Aerosócio-Ouro", "Aerosócio-Prata", "Rifa Avulsa"],
            "Valor": [500.0, 300.0, 50.0],
        }
    )
    demo["Hash_SHA256"] = demo["Doador"].apply(hash_sha256)
    st.dataframe(demo, use_container_width=True, hide_index=True)

    st.info("💡 Transações com k < 5 são automaticamente mascaradas.")

# ==================== TAB 5: PROFILING (LEGADO) ====================
with tab5:
    st.header("🔍 Data Profiling — Visão Raw")
    st.markdown(
        "Diagnóstico de qualidade dos dados preservado da versão anterior do dashboard."
    )

    st.subheader("Estatísticas Consolidadas (Jan–Ago 2026)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Gastos", f"R$ {df_trans[df_trans['Tipo']=='Despesa']['Valor_Abs'].sum():,.2f}")
    c2.metric("Total de Receitas", f"R$ {df_trans[df_trans['Tipo']=='Receita']['Valor'].sum():,.2f}")
    c3.metric("Nº de Transações", f"{len(df_trans)}")
    c4.metric(
        "Maior Saída Única",
        f"R$ {df_trans[df_trans['Tipo']=='Despesa']['Valor_Abs'].max():,.2f}",
    )

    with st.expander("📋 Extrato Completo (Camada Silver)"):
        df_show = df_trans.copy()
        df_show["Data"] = df_show["Data"].dt.strftime("%d/%m/%Y")
        st.dataframe(
            df_show.sort_values("Data", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
