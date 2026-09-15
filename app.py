import streamlit as st
import pandas as pd

st.title("Painel de Gastos Harpia 2026")

@st.cache_data
def carregar_dados():
    # URL RAW já configurada no seu repositório
    url = "https://raw.githubusercontent.com/yuriiih/HarpiaAerodesign_infraestrutura_dados/main/gastos%20harpia%202026.xlsx"
    df_raw = pd.read_excel(url, header=None)
    
    # Localiza o cabeçalho
    header_row_idx = 0
    for i, row in df_raw.iterrows():
        if 'Gastos' in row.values and 'Dias' in row.values:
            header_row_idx = i
            break
            
    df_temp = df_raw.iloc[header_row_idx + 1:].reset_index(drop=True)
    df_temp.columns = df_raw.iloc[header_row_idx]
    
    df_consolidado = df_temp[['Dias', 'Gastos']].dropna(subset=['Dias']).copy()
    df_consolidado.columns = ['Data', 'Valor']
    df_consolidado['Valor'] = pd.to_numeric(df_consolidado['Valor'], errors='coerce').fillna(0)
    
    return df_consolidado

df = carregar_dados()
st.dataframe(df)
st.write(f"**Total de Gastos:** R$ {df['Valor'].sum():.2f}")
