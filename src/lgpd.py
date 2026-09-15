# src/lgpd.py
"""
Governança de Dados: SHA-256 e k-Anonimato
"""
import hashlib
import pandas as pd

K_ANONIMATO = 5


def hash_sha256(valor: str) -> str:
    return hashlib.sha256(str(valor).encode()).hexdigest()


def anonimizar_doadores(df: pd.DataFrame, col_id: str) -> pd.DataFrame:
    """Aplica SHA-256 em identificadores de doadores."""
    df = df.copy()
    if col_id in df.columns:
        df[f'{col_id}_hash'] = df[col_id].apply(hash_sha256)
    return df


def aplicar_k_anonimato(df: pd.DataFrame, col_grupo: str, col_valor: str) -> pd.DataFrame:
    """
    Mascara valores quando o grupo tem menos de k registros.
    """
    df = df.copy()
    contagem = df.groupby(col_grupo)[col_valor].transform('count')
    df[col_valor + '_mascarado'] = df[col_valor].where(
        contagem >= K_ANONIMATO, "█████ (k < 5)"
    )
    return df
