# src/scenarios.py
# Substitua TODO o conteúdo por este.
"""
Simulação Estocástica com tratamento de outlier (aporte inicial único).
Versão 2.0 — Corrige inflação de σ_receita causada pelo aporte "Sobra 2025".
"""
import pandas as pd
import numpy as np
from typing import Dict

PISO_LIQUIDEZ = 5000.0
HORIZONTE_MESES = 6


def _stats_robustos(gold_data: dict) -> Dict[str, float]:
    """
    Recalcula μ e σ removendo o outlier do aporte inicial de Janeiro/2026
    (Sobra 2025 = R$ 6.898,56) que distorce a variabilidade histórica.
    """
    monthly = gold_data['monthly']
    trans   = gold_data['transactions']

    # Receita "recorrente" = receita total - aporte único inicial
    # Detecta: primeiro mês, tipo Receita, com descrição contendo 'sobra'
    aporte = trans[
        (trans['Tipo'] == 'Receita') &
        (trans['Descricao'].str.lower().str.contains('sobra|aporte inicial', na=False))
    ]['Valor_Abs'].sum()

    receitas_normais = monthly['Receita'].copy()
    if aporte > 0 and len(receitas_normais) > 0:
        receitas_normais.iloc[0] = max(receitas_normais.iloc[0] - aporte, 0.0)

    mu_r  = float(receitas_normais.mean())
    sig_r = float(receitas_normais.std(ddof=1)) if len(receitas_normais) > 1 else mu_r * 0.2
    mu_d  = float(monthly['Despesa'].mean())
    sig_d = float(monthly['Despesa'].std(ddof=1)) if len(monthly) > 1 else mu_d * 0.2

    return {
        'mu_receita': mu_r, 'sigma_receita': sig_r,
        'mu_despesa': mu_d, 'sigma_despesa': sig_d,
        'aporte_inicial_isolado': aporte,
    }


def simulate_scenarios(
    stats: Dict[str, float],
    saldo_atual: float,
    horizon: int = HORIZONTE_MESES,
) -> pd.DataFrame:
    """Projeta 3 cenários estocásticos baseados em μ ± σ."""
    mu_r  = stats['mu_receita']
    sig_r = stats['sigma_receita'] if stats['sigma_receita'] > 0 else mu_r * 0.2
    mu_d  = stats['mu_despesa']
    sig_d = stats['sigma_despesa'] if stats['sigma_despesa'] > 0 else mu_d * 0.2

    fluxo_conservador = (mu_r - 1.5 * sig_r) - (mu_d + 1.5 * sig_d)
    fluxo_moderado    = mu_r - mu_d
    fluxo_agressivo   = (mu_r + 1.0 * sig_r) - (mu_d - 0.5 * sig_d)

    datas = pd.date_range(
        start=pd.Timestamp.now().normalize().replace(day=1) + pd.DateOffset(months=1),
        periods=horizon, freq='MS',
    )
    out = pd.DataFrame({'Data': datas})
    out['Conservador'] = saldo_atual + np.arange(1, horizon + 1) * fluxo_conservador
    out['Moderado']    = saldo_atual + np.arange(1, horizon + 1) * fluxo_moderado
    out['Agressivo']   = saldo_atual + np.arange(1, horizon + 1) * fluxo_agressivo
    return out.set_index('Data')


def compute_risk_metrics(scenarios_df: pd.DataFrame) -> Dict:
    metrics = {}
    for cenario in ['Conservador', 'Moderado', 'Agressivo']:
        s = scenarios_df[cenario]
        metrics[cenario] = {
            'saldo_minimo': float(s.min()),
            'mes_critico':  s.idxmin().strftime('%b/%Y') if len(s) else 'N/A',
            'rompe_piso':   bool((s < PISO_LIQUIDEZ).any()),
            'meses_acima_piso': int((s >= PISO_LIQUIDEZ).sum()),
        }
    return metrics