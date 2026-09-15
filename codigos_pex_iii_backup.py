# -- coding: utf-8 --
"""Dashboard Harpia AeroDesign 2026
Código atualizado com interface Streamlit e correção de caminho de arquivo.
"""
import streamlit as st
import pandas as pd
import os
import logging
import plotly.express as px

# 1. Configuração da Página do Dashboard
st.set_page_config(
    page_title="Dashboard Harpia 2026",
    page_icon="🦅",
    layout="wide"
)

# Configuração de Logging (aparece nos logs do servidor do Streamlit)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- FUNÇÕES DE DATA PROFILING (Baseado no seu código original) ---
def profilename_financeiro(df):
    logging.info("Iniciando Data Profiling especializado...")
    if df.empty:
        st.warning("O DataFrame extraído está vazio.")
        return df
        
    nulos_valor = df['Valor'].isnull().sum()
    if nulos_valor > 0:
        st.warning(f"⚠️ Detectados {nulos_valor} registros sem valor.")
        
    datas_invalidas = df['Data'].isnull().sum()
    if datas_invalidas > 0:
        st.warning(f"⚠️ Detectadas {datas_invalidas} datas inválidas.")
        
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        st.info(f"ℹ️ Transações duplicadas identificadas: {duplicados}")
        
    return df

def profilename_financeiro_final(df):
    logging.info("--- Iniciando Data Profiling do Fluxo de Caixa (Harpia 2026) ---")
    
    nulos_valor = df['Valor'].isnull().sum()
    if nulos_valor > 0:
        st.error(f"❌ Alerta: Detectados {nulos_valor} registros sem valor monetário.")
    else:
        st.success("✅ Sucesso: Nenhum valor monetário nulo encontrado.")
        
    datas_invalidas = df['Data'].isnull().sum()
    if datas_invalidas > 0:
        st.error(f"❌ Erro: Inconsistência de tipo em {datas_invalidas} datas. Requer normalização ETL.")
    else:
        st.success(f"✅ Sucesso: Todas as {len(df)} datas foram validadas corretamente.")
        
    duplicados = df.duplicated(subset=['Data', 'Descricao', 'Valor']).sum()
    if duplicados > 0:
        st.info(f"ℹ️ Aviso: Transações duplicadas identificadas: {duplicados}")
    else:
        st.success("✅ Sucesso: Nenhuma transação duplicada encontrada.")
        
    return df

# --- FUNÇÃO DE CARGA E TRATAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    # 1. Definição blindada do caminho do arquivo
    NOME_ARQUIVO = "gastos harpia 2026.xlsx" 
    
    # Pega o diretório onde este script .py está rodando no servidor
    # Isso resolve o problema do Streamlit Cloud não achar o arquivo
    diretorio_script = os.path.dirname(os.path.abspath(__file__))
    caminho_completo = os.path.join(diretorio_script, NOME_ARQUIVO)
    
    # 2. Validação de Segurança
    if not os.path.exists(caminho_completo):
        st.error(f"❌ ERRO CRÍTICO: O Streamlit não achou o arquivo '{NOME_ARQUIVO}'.")
        st.warning(f"Caminho tentado no servidor: `{caminho_completo}`")
        st.info("👉 **O que fazer:** Vá no seu GitHub e verifique se o arquivo foi commitado na branch `main` e se o nome é *exatamente* igual (Linux diferencia maiúsculas de minúsculas!).")
        st.stop()
        
    # 3. Leitura da Planilha
    try:
        df_raw = pd.read_excel(caminho_completo, header=None)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel (Certifique-se que 'openpyxl' está no requirements.txt): {e}")
        st.stop()
        
    # Localiza a linha que contém os cabeçalhos 'Dias' e 'Gastos'
    header_row_idx = 0
    for i, row in df_raw.iterrows():
        if 'Gastos' in row.values and 'Dias' in row.values:
            header_row_idx = i
            break
            
    # Extração e Limpeza
    df_temp = df_raw.copy()
    df_temp.columns = df_raw.iloc[header_row_idx]
    df_temp = df_temp.iloc[header_row_idx + 1:].reset_index(drop=True)
    
    col_data = 'Dias'
    col_valor = 'Gastos'
    
    if col_data in df_temp.columns and col_valor in df_temp.columns:
        df_consolidado = df_temp[[col_data, col_valor]].copy()
        df_consolidado = df_consolidado.dropna(subset=[col_data])
        df_consolidado.columns = ['Data', 'Valor']
        df_consolidado['Descricao'] = "Transação Identificada"
        
        # Tratamento de Data (Baseado no seu código original - assumindo Jan/2026)
        # Adicionei uma proteção extra para converter floats (ex: 1.0) para inteiros (1)
        df_consolidado['Data'] = pd.to_datetime(
            df_consolidado['Data'].apply(lambda x: f"2026-01-{int(float(x)):02d}" if pd.notnull(x) and str(x).replace('.','',1).isdigit() else None), 
            errors='coerce'
        )
        df_consolidado = df_consolidado.dropna(subset=['Data'])
        
        # Garantindo que Valor seja numérico
        df_consolidado['Valor'] = pd.to_numeric(df_consolidado['Valor'], errors='coerce').fillna(0)
        
        return df_consolidado
    else:
        st.error(f"Colunas não encontradas. Disponíveis: {df_temp.columns.tolist()}")
        st.stop()

# --- INTERFACE DO DASHBOARD ---
st.title("🦅 Dashboard Financeiro - Harpia AeroDesign 2026")
st.markdown("Acompanhamento em tempo real do fluxo de caixa, infraestrutura e gastos da equipe.")
st.markdown("---")

# Carrega os dados
df = carregar_dados()

if not df.empty:
    # Execução dos profilings (adaptados para a tela do Streamlit)
    with st.expander("🔍 Ver Diagnóstico de Qualidade dos Dados (Data Profiling)"):
        df = profilename_financeiro(df)
        df = profilename_financeiro_final(df)
        
    st.markdown("---")
    
    # KPIs (Métricas Principais)
    st.subheader("📊 Visão Geral")
    col1, col2, col3, col4 = st.columns(4)
    
    total_gastos = df['Valor'].sum()
    maior_gasto = df['Valor'].max()
    ticket_medio = df['Valor'].mean()
    total_transacoes = len(df)
    
    col1.metric(label="Total de Gastos", value=f"R$ {total_gastos:,.2f}")
    col2.metric(label="Maior Gasto Único", value=f"R$ {maior_gasto:,.2f}")
    col3.metric(label="Ticket Médio", value=f"R$ {ticket_medio:,.2f}")
    col4.metric(label="Nº de Transações", value=f"{total_transacoes}")
    
    st.markdown("---")
    
    # Gráficos
    st.subheader("📈 Análise Temporal")
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
        st.markdown("**Evolução dos Gastos por Dia**")
        fig_linha = px.line(df, x='Data', y='Valor', markers=True, title="Fluxo de Caixa Diário")
        fig_linha.update_layout(xaxis_title="Data", yaxis_title="Valor (R$)")
        st.plotly_chart(fig_linha, use_container_width=True)
        
    with col_graf2:
        st.markdown("**Distribuição de Gastos**")
        fig_barra = px.bar(df, x='Data', y='Valor', title="Gastos por Data")
        fig_barra.update_layout(xaxis_title="Data", yaxis_title="Valor (R$)")
        st.plotly_chart(fig_barra, use_container_width=True)
        
    st.markdown("---")
    
    # Tabela de Dados
    st.subheader("📋 Extrato Detalhado")
    df_exibicao = df.copy()
    df_exibicao['Data'] = df_exibicao['Data'].dt.strftime('%d/%m/%Y')
    df_exibicao = df_exibicao.rename(columns={'Data': 'Data', 'Valor': 'Valor (R$)'})
    
    st.dataframe(
        df_exibicao[['Data', 'Valor (R$)']].sort_values('Data', ascending=False), 
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("Nenhum dado válido foi encontrado para exibição.")
