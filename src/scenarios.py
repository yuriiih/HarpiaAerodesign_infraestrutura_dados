# src/scenarios.py
"""
Simulação Estocástica: Conservador, Moderado e Agressivo
Baseado nas fórmulas do relatório técnico.
"""
import pandas as pd
import numpy as np
from typing import Dict

PISO_LIQUIDEZ = 5000.0
HORIZONTE_MESES = 6


def simulate_scenarios(
    stats: Dict[str, float],
    saldo_atual: float,
    horizon: int = HORIZONTE_MESES
) -> pd.DataFrame:
    """
    Projeta 3 cenários estocásticos baseados em μ ± σ.
    
    Fórmulas (do relatório):
    - Conservador: Y = μ_rec - 1.5σ_rec - (μ_gasto + 1.5σ_gasto)
    - Moderado:    Y = μ_rec - μ_gasto
    - Agressivo:   Y = μ_rec + 1.0σ_rec - (μ_gasto - 0.5σ_gasto)
    """
    mu_r = stats['mu_receita']
    sigma_r = stats['sigma_receita'] if stats['sigma_receita'] > 0 else mu_r * 0.2
    mu_d = stats['mu_despesa']
    sigma_d = stats['sigma_despesa'] if stats['sigma_despesa'] > 0 else mu_d * 0.2
    
    # Fluxo líquido mensal por cenário
    fluxo_conservador = (mu_r - 1.5 * sigma_r) - (mu_d + 1.5 * sigma_d)
    fluxo_moderado = mu_r - mu_d
    fluxo_agressivo = (mu_r + 1.0 * sigma_r) - (mu_d - 0.5 * sigma_d)
    
    datas = pd.date_range(
        start=pd.Timestamp.now().normalize().replace(day=1) + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'
    )
    
    resultado = pd.DataFrame({'Data': datas})
    resultado['Conservador'] = saldo_atual + np.arange(1, horizon + 1) * fluxo_conservador
    resultado['Moderado'] = saldo_atual + np.arange(1, horizon + 1) * fluxo_moderado
    resultado['Agressivo'] = saldo_atual + np.arange(1, horizon + 1) * fluxo_agressivo
    
    # Alertas de piso
    resultado['Alerta_Conservador'] = resultado['Conservador'] < PISO_LIQUIDEZ
    resultado['Alerta_Moderado'] = resultado['Moderado'] < PISO_LIQUIDEZ
    resultado['Alerta_Agressivo'] = resultado['Agressivo'] < PISO_LIQUIDEZ
    
    resultado = resultado.set_index('Data')
    return resultado


def compute_risk_metrics(scenarios_df: pd.DataFrame) -> Dict:
    """Calcula métricas de risco para exibição no dashboard."""
    metrics = {}
    for cenario in ['Conservador', 'Moderado', 'Agressivo']:
        serie = scenarios_df[cenario]
        metrics[cenario] = {
            'saldo_minimo': serie.min(),
            'mes_critico': serie.idxmin().strftime('%b/%Y') if len(serie) > 0 else 'N/A',
            'rompe_piso': bool((serie < PISO_LIQUIDEZ).any()),
            'meses_acima_piso': int((serie >= PISO_LIQUIDEZ).sum()),
            'probabilidade_ruina': f"{(serie < PISO_LIQUIDEZ).sum() / len(serie) * 100:.0f}%",
        }
    return metrics
