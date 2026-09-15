import streamlit as st
import pandas as pd
import os
import plotly.express as px

# 1. Configuração da Página do Dashboard
st.set_page_config(
    page_title="Dashboard Harpia 2026",
    page_icon="🦅",
    layout="wide"
)

# 2. Função de Carga e Tratamento de Dados (Com cache para não recarregar a cada clique)
@st.cache_data
def carregar_e_tratar_dados(nome_arquivo):
    # --- CORREÇÃO DO FileNotFoundError ---
    # Busca o arquivo na raiz ou em subpastas comuns para evitar erros de CWD no Streamlit Cloud
    file_path = nome_arquivo
    if not os.path.exists(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        alt_path = os.path.join(script_dir, nome_arquivo)
        
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            for folder in ['data', 'dados', '']:
                folder_path = os.path.join(script_dir, folder, nome_arquivo)
                if os.path.exists(folder_path):
                    file_path = folder_path
                    break
            
            if not os.path.exists(file_path):
                st.error(f"❌ ERRO CRÍTICO: O arquivo '{nome_arquivo}' não foi encontrado no repositório.")
                st.info("AÇÃO: Verifique se você deu 'git add' e 'git commit' na planilha e se ela não está sendo ignorada pelo '.gitignore'.")
                st.stop()

    # Leitura e Parsing da Planilha
    try:
        df_raw = pd.read_excel(file_path, header=None)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel (Verifique se o openpyxl está instalado): {e}")
        st.stop()

    header_row_idx = 0
    for i, row in df_raw.iterrows():
        if 'Gastos' in row.values and 'Dias' in row.values:
            header_row_idx = i
            break
            
    df_temp = df_raw.iloc[header_row_idx + 1:].copy()
    df_temp.columns = df_raw.iloc[header_row_idx]
    df_temp = df_temp.reset_index(drop=True)
    
    if 'Dias' not in df_temp.columns or 'Gastos' not in df_temp.columns:
        st.error("As colunas 'Dias' e 'Gastos' não foram encontradas na planilha.")
        st.stop()

    # Limpeza e Consolidação
    df = df_temp[['Dias', 'Gastos']].dropna(subset=['Dias']).copy()
    df.columns = ['Dia', 'Valor']
    df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0.0)
    
    # Criação da Data (Assumindo Janeiro de 2026 com base no contexto)
    df['Data'] = pd.to_datetime(
        df['Dia'].apply(lambda x: f"2026-01-{int(x):02d}" if str(x).isdigit() else None), 
        errors='coerce'
    )
    df = df.dropna(subset=['Data'])
    df = df.sort_values('Data')
    
    return df

# 3. Interface do Dashboard
st.title("🦅 Dashboard Financeiro - Harpia AeroDesign 2026")
st.markdown("Acompanhamento em tempo real do fluxo de caixa, infra-estrutura e gastos da equipe.")
st.markdown("---")

# Carrega os dados
df = carregar_e_tratar_dados("gastos harpia 2026.xlsx")

if not df.empty:
    # --- KPIs (Métricas Principais no Topo) ---
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
    
    # --- Gráficos Interativos ---
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
    
    # --- Tabela de Dados (Extrato) ---
    st.subheader("📋 Extrato Detalhado")
    df_exibicao = df.copy()
    df_exibicao['Data'] = df_exibicao['Data'].dt.strftime('%d/%m/%Y')
    df_exibicao = df_exibicao.rename(columns={'Dia': 'Dia do Mês', 'Valor': 'Valor (R$)'})
    
    st.dataframe(
        df_exibicao[['Data', 'Valor (R$)']].sort_values('Data', ascending=False), 
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("Nenhum dado válido foi encontrado para exibição.")
