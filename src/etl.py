# src/etl.py
# -*- coding: utf-8 -*-
"""
Pipeline ETL — Arquitetura Medallion (Bronze → Silver → Gold)
Versão 3.2 — parser à prova do layout real da planilha Harpia 2026.
"""
import os
import re
import logging
import unicodedata

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MESES = {'janeiro':1,'fevereiro':2,'marco':3,'abril':4,'maio':5,'junho':6,
         'julho':7,'agosto':8,'setembro':9,'outubro':10,'novembro':11,'dezembro':12}


def _norm(s):
    s = unicodedata.normalize('NFKD', str(s).lower())
    return ''.join(c for c in s if not unicodedata.combining(c)).strip()


def _num(x):
    return pd.to_numeric(pd.Series([x]), errors='coerce').iloc[0]


# ==================== BRONZE ====================
def load_bronze(fonte):
    if isinstance(fonte, dict):
        sheets = fonte
    else:
        if not isinstance(fonte, (str, bytes, os.PathLike)):
            raise TypeError(f"Esperado dict/caminho/URL, recebido {type(fonte).__name__}")
        sheets = pd.read_excel(fonte, sheet_name=None, header=None)
    out = {}
    for nome, df in sheets.items():
        k = _norm(nome)
        if k in MESES:
            out[MESES[k]] = df
    if not out:
        raise ValueError(f"Nenhuma aba de mês encontrada. Abas: {list(sheets)}")
    logger.info(f"Bronze: {len(out)} aba(s) de mês carregada(s).")
    return out


# ==================== PARSERS ====================
def _find_label_row(df, labels, max_rows=25):
    """Primeira linha (entre as 25 primeiras) contendo TODOS os rótulos."""
    want = {_norm(l) for l in labels}
    for r in range(min(max_rows, len(df))):
        cells = {_norm(v) for v in df.iloc[r] if pd.notnull(v)}
        if want <= cells:
            return r
    return None


def _find_dias_column(df):
    """Fallback: coluna cuja sequência vertical é 1,2,3,..."""
    for c in range(df.shape[1]):
        col = pd.to_numeric(df.iloc[:, c], errors='coerce').dropna()
        if len(col) < 28:
            continue
        vals = col.head(31).tolist()
        if vals == list(range(1, len(vals) + 1)):
            return c
    return None


def _parse_diario(df, mes):
    i_d = i_g = i_s = None
    hdr = _find_label_row(df, ['Dias', 'Gastos'])
    if hdr is not None:
        rot = {}
        for i, v in enumerate(df.iloc[hdr]):
            if pd.notnull(v):
                rot.setdefault(_norm(v), i)
        i_d, i_g, i_s = rot.get('dias'), rot.get('gastos'), rot.get('saldo')

    if i_d is None or i_g is None:                      # fallback estrutural
        i_d = _find_dias_column(df)
        if i_d is not None:
            cand = [c for c in (i_d - 3, i_d - 2, i_d - 1) if 0 <= c < df.shape[1]]
            zeros = {}
            for c in cand:
                col = pd.to_numeric(df.iloc[:, c], errors='coerce').fillna(0.0)
                zeros[c] = int((col == 0).sum())
            i_g = max(zeros, key=zeros.get) if zeros else None
            resto = [c for c in cand if c != i_g]
            i_s = resto[0] if resto else None

    if i_d is None or i_g is None:
        raise ValueError(f"Aba mês {mes}: bloco diário não localizado nem por rótulo nem por estrutura.")

    regs, seen = [], set()
    for r in range(1, len(df)):
        if str(df.iat[r, 0]).startswith('Progressão'):
            continue
        d = _num(df.iat[r, i_d])
        if pd.notnull(d) and 1 <= d <= 31 and float(d) == int(d):
            d = int(d)
            if d in seen:
                continue
            seen.add(d)
            g = _num(df.iat[r, i_g])
            s = _num(df.iat[r, i_s]) if i_s is not None else np.nan
            regs.append((d, float(g or 0.0), s))
    return pd.DataFrame(regs, columns=['dia', 'gasto', 'saldo'])


def _parse_transacoes(df):
    i_hdr = None
    for r in range(len(df)):
        vals = {_norm(v) for v in df.iloc[r] if pd.notnull(v)}
        if 'dia 1' in vals and 'dia 2' in vals:
            i_hdr = r
            break
    if i_hdr is None:
        return []
    pos = {}
    for i, v in enumerate(df.iloc[i_hdr]):
        if pd.notnull(v):
            m = re.fullmatch(r'dia\s+(\d{1,2})', _norm(v))
            if m:
                pos.setdefault(int(m.group(1)), i)
    inicio = i_hdr + 2
    for r in range(i_hdr + 1, min(i_hdr + 6, len(df))):
        vals = {_norm(v) for v in df.iloc[r] if pd.notnull(v)}
        if 'entradas' in vals and 'saidas' in vals:
            inicio = r + 1
            break
    trans = []
    for r in range(inicio, len(df)):
        if str(df.iat[r, 0]).startswith('Progressão'):
            break
        for dia, p in pos.items():
            for off, tipo in ((0, 'Receita'), (2, 'Despesa')):
                if p + off + 1 >= df.shape[1]:
                    continue
                v = _num(df.iat[r, p + off])
                desc = df.iat[r, p + off + 1]
                if pd.notnull(v) and abs(float(v)) > 1e-9:
                    trans.append((dia, tipo,
                                  str(desc).strip() if pd.notnull(desc) else 'Sem descrição',
                                  abs(float(v))))
    return trans


def _parse_resumo(df):
    """
    Localiza a célula 'Faturamento' em QUALQUER coluna das primeiras 25 linhas.
    O valor fica na mesma coluna da linha imediatamente abaixo.
    Idem para 'Gastos' (do Resumo, não do bloco diário).
    """
    fat_col = fat_row = gasto_col = gasto_row = None
    for r in range(min(25, len(df))):
        for c in range(df.shape[1]):
            v = df.iat[r, c]
            if pd.notnull(v):
                n = _norm(v)
                if n == 'faturamento' and fat_row is None:
                    fat_row, fat_col = r, c
                elif n == 'gastos' and gasto_row is None and fat_row is not None:
                    # pega o primeiro 'Gastos' que aparece APÓS Faturamento
                    # (para não confundir com o rótulo do bloco diário)
                    gasto_row, gasto_col = r, c
    fat_val = gasto_val = None
    if fat_row is not None and fat_row + 1 < len(df):
        # tenta a mesma coluna; se for vazia, tenta a próxima
        for cc in (fat_col, fat_col + 1):
            if cc < df.shape[1]:
                v = _num(df.iat[fat_row + 1, cc])
                if pd.notnull(v) and float(v) != 0:
                    fat_val = float(v)
                    break
    if gasto_row is not None and gasto_row + 1 < len(df):
        for cc in (gasto_col, gasto_col + 1):
            if cc < df.shape[1]:
                v = _num(df.iat[gasto_row + 1, cc])
                if pd.notnull(v) and float(v) != 0:
                    gasto_val = float(v)
                    break
    return {'faturamento': fat_val, 'gastos': gasto_val}


def _categorizar(desc, tipo):
    d = str(desc).lower()
    if tipo == 'Receita':
        if 'rifa' in d: return 'Rifa'
        if 'aerosocio' in d or 'aerosócio' in d: return 'Aerosócio'
        if 'sobra' in d: return 'Aporte Inicial'
        if 'juros' in d: return 'Rendimentos'
        if 'churrasco' in d or 'choconhaque' in d or 'shot' in d or 'chocolate' in d or 'festa' in d: return 'Eventos'
        if 'lojinha' in d: return 'Lojinha'
        return 'Outras Receitas'
    if 'motor' in d or 'servo' in d or 'bateria' in d or 'helice' in d or 'switch' in d or 'conector' in d or 'sensor' in d or 'telemetria' in d:
        return 'Eletrônica/Propulsão'
    if 'mdf' in d or 'usinag' in d or 'molde' in d or 'cera' in d or 'madeira' in d or 'isopor' in d or 'balsa' in d:
        return 'Estrutura'
    if 'resina' in d or 'lamina' in d or 'peel' in d or 'manta' in d or 'tekbond' in d or 'cola' in d or 'entelagem' in d or 'fibra' in d:
        return 'Laminação'
    if 'drive' in d or 'assinatura' in d or 'taxa' in d or 'licença' in d or 'licenca' in d or 'nuvem' in d or 'mensalidade' in d:
        return 'Custos Fixos'
    return 'Outros Gastos'


# ==================== SILVER ====================
def silver_clean(bronze_sheets, ano=2026):
    rows, checks, falhas = [], [], []
    for mes, df in sorted(bronze_sheets.items()):
        try:
            di  = _parse_diario(df, mes)
            tr  = _parse_transacoes(df)
            res = _parse_resumo(df)
        except Exception as e:
            logger.warning(f"Aba do mês {mes} PULADA: {e}")
            falhas.append(mes)
            continue
        for dia, tipo, desc, v in tr:
            rows.append(dict(Data=pd.Timestamp(year=ano, month=mes, day=dia),
                             Mes=mes, Tipo=tipo, Descricao=desc, Valor_Abs=v))
        sf = di['saldo'].dropna()
        checks.append(dict(mes=mes,
                           gasto_diario=float(di['gasto'].sum()),
                           saidas=sum(v for _, t, _, v in tr if t == 'Despesa'),
                           entradas=sum(v for _, t, _, v in tr if t == 'Receita'),
                           resumo_fat=res.get('faturamento'),
                           resumo_gasto=res.get('gastos'),
                           saldo_final=float(sf.iloc[-1]) if len(sf) else np.nan))
    if not rows:
        raise ValueError("Nenhuma transação extraída de nenhuma aba. Verifique os WARNINGs acima.")
    silver = pd.DataFrame(rows)
    silver['Categoria'] = silver.apply(lambda r: _categorizar(r['Descricao'], r['Tipo']), axis=1)
    silver['Valor'] = np.where(silver['Tipo'] == 'Receita', silver['Valor_Abs'], -silver['Valor_Abs'])
    if falhas:
        logger.warning(f"Abas não processadas: {falhas}")
    logger.info(f"Silver: {len(silver)} transações tipadas.")
    return silver, pd.DataFrame(checks)


# ==================== GOLD ====================
def gold_aggregate(silver, checks):
    df = silver.copy()
    rec  = df[df['Tipo'] == 'Receita'].groupby('Mes')['Valor_Abs'].sum()
    desp = df[df['Tipo'] == 'Despesa'].groupby('Mes')['Valor_Abs'].sum()
    gold = pd.DataFrame({'Receita': rec, 'Despesa': desp}).fillna(0.0)
    gold = gold[(gold['Receita'] > 0) | (gold['Despesa'] > 0)]
    gold['Saldo_Mes'] = gold['Receita'] - gold['Despesa']
    gold['Saldo_Acumulado'] = gold['Saldo_Mes'].cumsum()
    gold.index = pd.PeriodIndex([f"2026-{m:02d}" for m in gold.index], freq='M').to_timestamp()
    n = len(gold)
    return {
        'monthly': gold,
        'transactions': df,
        'checks': checks,
        'stats': {
            'mu_receita':    float(gold['Receita'].mean())  if n else 0.0,
            'sigma_receita': float(gold['Receita'].std())   if n > 1 else 0.0,
            'mu_despesa':    float(gold['Despesa'].mean())  if n else 0.0,
            'sigma_despesa': float(gold['Despesa'].std())   if n > 1 else 0.0,
        },
    }


def run_pipeline(fonte, ano=2026):
    bronze = load_bronze(fonte)
    silver, checks = silver_clean(bronze, ano=ano)
    return gold_aggregate(silver, checks)