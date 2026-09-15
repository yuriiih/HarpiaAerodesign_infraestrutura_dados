# src/etl.py
"""
Pipeline ETL - Arquitetura Medallion (Bronze → Silver → Gold)
"""
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== BRONZE ====================
def load_bronze(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega a planilha bruta e identifica cabeçalhos dinamicamente."""
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    df_raw = pd.read_excel(caminho_arquivo, header=None)
    
    # Detecta linha de cabeçalho
    header_idx = 0
    for i, row in df_raw.iterrows():
        row_str = [str(v).lower() for v in row.values if pd.notnull(v)]
        if any('data' in s or 'dia' in s for s in row_str) and \
           any('valor' in s or 'gasto' in s or 'receita' in s for s in row_str):
            header_idx = i
            break
    
    df = df_raw.copy()
    df.columns = df_raw.iloc[header_idx]
    df = df.iloc[header_idx + 1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    
    logger.info(f"Bronze: {len(df)} registros carregados")
    return df


# ==================== SILVER ====================
def silver_clean(df_bronze: pd.DataFrame) -> pd.DataFrame:
    """Limpeza, tipagem e categorização."""
    df = df_bronze.copy()
    
    # Identifica colunas de data e valor
    col_data = next((c for c in df.columns if 'data' in c.lower() or 'dia' in c.lower()), None)
    col_valor = next((c for c in df.columns if 'valor' in c.lower() or 'gasto' in c.lower()), None)
    col_desc = next((c for c in df.columns if 'desc' in c.lower() or 'categ' in c.lower()), None)
    col_tipo = next((c for c in df.columns if 'tipo' in c.lower()), None)
    
    if not col_data or not col_valor:
        raise ValueError(f"Colunas essenciais não encontradas. Disponíveis: {df.columns.tolist()}")
    
    # Normaliza data
    df['Data'] = pd.to_datetime(df[col_data], errors='coerce')
    
    # Se a coluna for apenas dia (numérico), assume 2026
    if df['Data'].isna().all():
        df['Data'] = df[col_data].apply(
            lambda x: pd.to_datetime(f"2026-01-{int(float(x)):02d}", errors='coerce')
            if pd.notnull(x) and str(x).replace('.', '').replace('-', '').isdigit()
            else pd.NaT
        )
    
    # Normaliza valor
    df['Valor'] = pd.to_numeric(df[col_valor], errors='coerce').fillna(0)
    
    # Descrição e Categoria
    df['Descricao'] = df[col_desc] if col_desc else "Transação"
    df['Tipo'] = df[col_tipo] if col_tipo else df['Valor'].apply(
        lambda v: 'Receita' if v > 0 else 'Despesa'
    )
    
    # Categoriza por subsistema
    def categorizar(desc, tipo):
        desc_l = str(desc).lower()
        if tipo == 'Receita':
            if 'rifa' in desc_l: return 'Rifa'
            if 'aerosócio' in desc_l or 'aerosocio' in desc_l: return 'Aerosócio'
            if 'evento' in desc_l or 'festa' in desc_l: return 'Evento'
            return 'Outras Receitas'
        else:
            if 'motor' in desc_l or 'servo' in desc_l or 'eletrôn' in desc_l or 'bateria' in desc_l:
                return 'Eletrônica/Propulsão'
            if 'mdf' in desc_l or 'usinag' in desc_l or 'molde' in desc_l or 'cera' in desc_l:
                return 'Estrutura'
            if 'fibra' in desc_l or 'resina' in desc_l or 'epoxy' in desc_l or 'lamina' in desc_l:
                return 'Laminação'
            if 'taxa' in desc_l or 'nuvem' in desc_l or 'banco' in desc_l or 'tarifa' in desc_l:
                return 'Custos Fixos'
            return 'Outros Gastos'
    
    df['Categoria'] = df.apply(lambda r: categorizar(r['Descricao'], r['Tipo']), axis=1)
    df['Valor_Abs'] = df['Valor'].abs()
    
    df = df.dropna(subset=['Data'])
    logger.info(f"Silver: {len(df)} registros limpos")
    return df[['Data', 'Descricao', 'Tipo', 'Categoria', 'Valor', 'Valor_Abs']]


# ==================== GOLD ====================
def gold_aggregate(df_silver: pd.DataFrame) -> dict:
    """Agrega para nível mensal — pronto para modelagem."""
    df = df_silver.copy()
    df['Mes'] = df['Data'].dt.to_period('M')
    
    # Receitas e despesas mensais
    receitas = df[df['Tipo'] == 'Receita'].groupby('Mes')['Valor'].sum()
    despesas = df[df['Tipo'] == 'Despesa'].groupby('Mes')['Valor_Abs'].sum()
    
    # Gastos por subsistema
    subsistemas = df[df['Tipo'] == 'Despesa'].groupby(['Mes', 'Categoria'])['Valor_Abs'].sum().unstack(fill_value=0)
    
    gold = pd.DataFrame({
        'Receita': receitas,
        'Despesa': despesas,
    }).fillna(0)
    
    gold['Saldo_Mes'] = gold['Receita'] - gold['Despesa']
    gold['Saldo_Acumulado'] = gold['Saldo_Mes'].cumsum()
    
    # Junta subsistemas
    gold = gold.join(subsistemas, how='left').fillna(0)
    gold.index = gold.index.to_timestamp()
    
    logger.info(f"Gold: {len(gold)} meses agregados")
    return {
        'monthly': gold,
        'transactions': df_silver,
        'stats': {
            'mu_receita': gold['Receita'].mean(),
            'sigma_receita': gold['Receita'].std(),
            'mu_despesa': gold['Despesa'].mean(),
            'sigma_despesa': gold['Despesa'].std(),
        }
    }


def run_pipeline(caminho: str) -> dict:
    """Executa o pipeline completo."""
    bronze = load_bronze(caminho)
    silver = silver_clean(bronze)
    gold = gold_aggregate(silver)
    return gold
