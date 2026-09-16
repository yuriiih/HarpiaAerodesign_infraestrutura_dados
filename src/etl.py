# src/etl.py
# -*- coding: utf-8 -*-
"""
Pipeline ETL — Arquitetura Medallion (Bronze → Silver → Gold)
Versão 2.0 (CORRIGIDA)

CORREÇÕES:
1. BUG CRÍTICO (stat: path should be string... not DataFrame):
   load_bronze()/run_pipeline() agora aceitam DataFrame EM MEMÓRIA
   (entregue pelo app.py após ler a URL raw) OU caminho/URL (str).
2. BUG LATENTE (datas caindo em 1970):
   pd.to_datetime() sobre inteiros (ex.: 15) interpreta como epoch.
   Agora detectamos colunas só com dias (1–31) e reconstruímos a data real.
3. Compatibilidade total com o layout original ('Dias'/'Gastos').
"""
import os
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CAMADA BRONZE — ingestão bruta
# ============================================================
def load_bronze(fonte) -> pd.DataFrame:
    """
    Recebe a fonte de dados e retorna o DataFrame bruto.

    fonte : pd.DataFrame | str
        - DataFrame: já carregado pelo app.py (URL raw ou arquivo local).
        - str: caminho local ou URL http(s) do .xlsx.
    """
    # --- Caso 1: DataFrame já em memória (CORREÇÃO DO BUG CRÍTICO) ---
    if isinstance(fonte, pd.DataFrame):
        logger.info(f"Bronze: DataFrame recebido em memória ({len(fonte)} linhas).")
        return fonte.copy()

    # --- Guarda de tipo: impede que o erro de stat() volte a acontecer ---
    if not isinstance(fonte, (str, bytes, os.PathLike)):
        raise TypeError(
            "load_bronze() espera caminho/URL (str) ou pandas.DataFrame, "
            f"mas recebeu: {type(fonte).__name__}"
        )

    # --- Caso 2: URL ou caminho local ---
    eh_url = str(fonte).startswith(("http://", "https://"))
    if not eh_url and not os.path.exists(fonte):
        raise FileNotFoundError(f"Arquivo não encontrado: {fonte}")

    df_raw = pd.read_excel(fonte, header=None)
    logger.info(f"Bronze: {len(df_raw)} linhas lidas de {'URL' if eh_url else 'arquivo local'}.")

    # --- Detecção do cabeçalho (compatível com seu código original) ---
    header_idx = 0
    for i, row in df_raw.iterrows():                      # 1º: busca exata
        vals = [str(v).strip() for v in row.values if pd.notnull(v)]
        if 'Dias' in vals and 'Gastos' in vals:
            header_idx = i
            break
    else:                                                  # 2º: busca flexível
        for i, row in df_raw.iterrows():
            cells = [str(v).strip().lower() for v in row.values if pd.notnull(v)]
            tem_data = any(c in ('dias', 'dia', 'data') or c.startswith('data') for c in cells)
            tem_valor = any(('valor' in c) or ('gasto' in c) or ('receita' in c) for c in cells)
            if tem_data and tem_valor:
                header_idx = i
                break

    df = df_raw.iloc[header_idx + 1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df_raw.iloc[header_idx]]
    return df


# ============================================================
# CAMADA SILVER — limpeza, tipagem e categorização
# ============================================================
def _categorizar(desc: str, tipo: str) -> str:
    d = str(desc).lower()
    if tipo == 'Receita':
        if 'rifa' in d: return 'Rifa'
        if 'aerosócio' in d or 'aerosocio' in d: return 'Aerosócio'
        if 'festa' in d or 'evento' in d: return 'Evento'
        return 'Outras Receitas'
    if 'motor' in d or 'servo' in d or 'eletrôn' in d or 'bateria' in d or 'esc' in d:
        return 'Eletrônica/Propulsão'
    if 'mdf' in d or 'usinag' in d or 'molde' in d or 'cera' in d:
        return 'Estrutura'
    if 'fibra' in d or 'resina' in d or 'epoxy' in d or 'lamina' in d or 'tecido' in d:
        return 'Laminação'
    if 'taxa' in d or 'nuvem' in d or 'banco' in d or 'tarifa' in d or 'manuten' in d:
        return 'Custos Fixos'
    return 'Outros Gastos'


def silver_clean(df_bronze: pd.DataFrame,
                 mes_padrao: int = 1,
                 ano_padrao: int = 2026) -> pd.DataFrame:
    df = df_bronze.copy()
    cols = [str(c) for c in df.columns]

    col_data  = next((c for c in cols if c.lower() in ('dias', 'dia', 'data') or 'data' in c.lower()), None)
    col_valor = next((c for c in cols if 'valor' in c.lower() or 'gasto' in c.lower()), None)
    col_desc  = next((c for c in cols if 'desc' in c.lower() or 'categ' in c.lower() or 'hist' in c.lower()), None)
    col_tipo  = next((c for c in cols if 'tipo' in c.lower()), None)
    col_mes   = next((c for c in cols if c.lower() in ('mes', 'mês')), None)

    if col_valor is None:
        raise ValueError(f"Coluna de valor não encontrada. Disponíveis: {cols}")

    # ---------- VALOR ----------
    df['Valor'] = pd.to_numeric(df[col_valor], errors='coerce').fillna(0.0)

    # ---------- DATA (com correção do bug de 1970) ----------
    if col_data is not None:
        serie = df[col_data]
        datas = pd.to_datetime(serie, errors='coerce', dayfirst=True)

        anos = datas.dt.year.dropna()
        if len(anos) > 0 and (anos < 2000).mean() > 0.5:
            # Valores 1–31 caíram no epoch (1970): reconstrói data real
            logger.warning("Coluna de data contém apenas dias (1–31). Reconstruindo datas completas.")
            dias = pd.to_numeric(serie, errors='coerce')
            meses = pd.to_numeric(df[col_mes], errors='coerce').fillna(mes_padrao) if col_mes else mes_padrao
            datas = pd.to_datetime(
                pd.DataFrame({'year': ano_padrao, 'month': meses, 'day': dias}),
                errors='coerce',
            )
    else:
        datas = pd.Series(pd.NaT, index=df.index)

    df['Data'] = datas

    # ---------- DESCRIÇÃO / TIPO ----------
    df['Descricao'] = df[col_desc].fillna('Transação Identificada') if col_desc else 'Transação Identificada'

    if col_tipo is not None:
        mapa = {'entrada': 'Receita', 'receita': 'Receita',
                'saida': 'Despesa', 'saída': 'Despesa', 'despesa': 'Despesa', 'gasto': 'Despesa'}
        df['Tipo'] = df[col_tipo].astype(str).str.lower().str.strip().map(lambda t: mapa.get(t, 'Receita'))
    else:
        nome_valor = col_valor.lower()
        if 'gasto' in nome_valor or 'despesa' in nome_valor:
            df['Tipo'] = np.where(df['Valor'] < 0, 'Receita', 'Despesa')
        elif 'receita' in nome_valor or 'entrada' in nome_valor:
            df['Tipo'] = np.where(df['Valor'] < 0, 'Despesa', 'Receita')
        else:
            df['Tipo'] = np.where(df['Valor'] >= 0, 'Receita', 'Despesa')

    df['Valor_Abs'] = df['Valor'].abs()
    df = df.dropna(subset=['Data'])
    df['Categoria'] = df.apply(lambda r: _categorizar(r['Descricao'], r['Tipo']), axis=1)

    logger.info(f"Silver: {len(df)} registros limpos e tipados.")
    return df[['Data', 'Descricao', 'Tipo', 'Categoria', 'Valor', 'Valor_Abs']]


# ============================================================
# CAMADA GOLD — agregação mensal para modelagem
# ============================================================
def gold_aggregate(df_silver: pd.DataFrame) -> dict:
    df = df_silver.copy()
    df['Mes'] = df['Data'].dt.to_period('M')

    receitas = df[df['Tipo'] == 'Receita'].groupby('Mes')['Valor'].sum()
    despesas = df[df['Tipo'] == 'Despesa'].groupby('Mes')['Valor_Abs'].sum()

    gold = pd.DataFrame({'Receita': receitas, 'Despesa': despesas}).fillna(0.0)
    gold['Saldo_Mes'] = gold['Receita'] - gold['Despesa']
    gold['Saldo_Acumulado'] = gold['Saldo_Mes'].cumsum()

    subs = (df[df['Tipo'] == 'Despesa']
            .groupby(['Mes', 'Categoria'])['Valor_Abs'].sum()
            .unstack(fill_value=0))
    gold = gold.join(subs, how='left').fillna(0.0)
    gold.index = gold.index.to_timestamp()
    gold = gold.sort_index()

    n = len(gold)
    logger.info(f"Gold: {n} mês(es) agregado(s).")
    return {
        'monthly': gold,
        'transactions': df_silver,
        'stats': {
            'mu_receita':    float(gold['Receita'].mean()) if n else 0.0,
            'sigma_receita': float(gold['Receita'].std())  if n > 1 else 0.0,
            'mu_despesa':    float(gold['Despesa'].mean()) if n else 0.0,
            'sigma_despesa': float(gold['Despesa'].std())  if n > 1 else 0.0,
        },
    }


# ============================================================
# ORQUESTRADOR (ACEITA DataFrame OU str)
# ============================================================
def run_pipeline(fonte, mes_padrao: int = 1, ano_padrao: int = 2026) -> dict:
    """Executa Bronze → Silver → Gold. `fonte` pode ser DataFrame ou caminho/URL."""
    bronze = load_bronze(fonte)
    silver = silver_clean(bronze, mes_padrao=mes_padrao, ano_padrao=ano_padrao)
    return gold_aggregate(silver)
