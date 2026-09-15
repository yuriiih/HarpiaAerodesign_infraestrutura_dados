# src/viz.py
"""
Gráficos Plotly reutilizáveis para o dashboard.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

PISO_LIQUIDEZ = 5000.0


def fig_historico_saldo(df_gold: pd.DataFrame) -> go.Figure:
    """Saldo acumulado histórico com zonas de fase."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_gold.index, y=df_gold['Saldo_Acumulado'],
        mode='lines+markers', name='Saldo Acumulado',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Linha do piso
    fig.add_hline(y=PISO_LIQUIDEZ, line_dash="dash", line_color="red",
                  annotation_text=f"Piso Liquidez (R$ {PISO_LIQUIDEZ:,.0f})")
    
    # Zonas de fase
    meses = df_gold.index.month
    for i, (start, end, color, label) in enumerate([
        (1, 4, 'rgba(0,200,0,0.08)', 'Acumulação'),
        (5, 7, 'rgba(255,0,0,0.08)', 'Desinvestimento'),
        (8, 12, 'rgba(128,128,128,0.08)', 'Estagnação'),
    ]):
        mask = (meses >= start) & (meses <= end)
        if mask.any():
            fig.add_vrect(
                x0=df_gold.index[mask].min(), x1=df_gold.index[mask].max(),
                fillcolor=color, line_width=0,
                annotation_text=label, annotation_position="top"
            )
    
    fig.update_layout(
        title="Evolução do Saldo Acumulado — Fases do Ciclo Anual",
        xaxis_title="Período", yaxis_title="Saldo (R$)",
        template="plotly_white", height=450
    )
    return fig


def fig_previsao_ensemble(df_historico: pd.DataFrame, df_forecast: pd.DataFrame) -> go.Figure:
    """Gráfico de previsão combinada com banda de confiança."""
    fig = go.Figure()
    
    # Histórico
    fig.add_trace(go.Scatter(
        x=df_historico.index, y=df_historico['Saldo_Acumulado'],
        mode='lines+markers', name='Histórico',
        line=dict(color='#1f77b4', width=2)
    ))
    
    if not df_forecast.empty and 'Ensemble_Pred' in df_forecast.columns:
        # Previsão
        fig.add_trace(go.Scatter(
            x=df_forecast.index, y=df_forecast['Ensemble_Pred'],
            mode='lines+markers', name='Previsão Ensemble',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        # Banda de confiança (se SARIMAX disponível)
        if 'SARIMAX_Lower' in df_forecast.columns:
            fig.add_trace(go.Scatter(
                x=df_forecast.index, y=df_forecast['SARIMAX_Upper'],
                mode='lines', name='IC 95% Superior',
                line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df_forecast.index, y=df_forecast['SARIMAX_Lower'],
                mode='lines', name='IC 95% Inferior',
                line=dict(width=0), fill='tonexty',
                fillcolor='rgba(255,127,14,0.15)', showlegend=True
            ))
    
    fig.add_hline(y=PISO_LIQUIDEZ, line_dash="dash", line_color="red",
                  annotation_text=f"Piso R$ {PISO_LIQUIDEZ:,.0f}")
    
    fig.update_layout(
        title="Previsão de Saldo — Ensemble (SARIMAX + XGBoost)",
        xaxis_title="Período", yaxis_title="Saldo Projetado (R$)",
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


def fig_cenarios_estocasticos(scenarios_df: pd.DataFrame) -> go.Figure:
    """3 cenários estocásticos com alertas de ruptura."""
    fig = go.Figure()
    
    cores = {'Conservador': '#d62728', 'Moderado': '#ff7f0e', 'Agressivo': '#2ca02c'}
    
    for cenario, cor in cores.items():
        fig.add_trace(go.Scatter(
            x=scenarios_df.index, y=scenarios_df[cenario],
            mode='lines+markers', name=cenario,
            line=dict(color=cor, width=2.5),
            marker=dict(size=7)
        ))
        
        # Destaca pontos abaixo do piso
        abaixo = scenarios_df[scenarios_df[cenario] < PISO_LIQUIDEZ]
        if not abaixo.empty:
            fig.add_trace(go.Scatter(
                x=abaixo.index, y=abaixo[cenario],
                mode='markers', name=f'⚠️ {cenario} (Risco)',
                marker=dict(color=cor, size=14, symbol='x-thin',
                           line=dict(width=2, color='black')),
                showlegend=False
            ))
    
    fig.add_hline(y=PISO_LIQUIDEZ, line_dash="dash", line_color="black",
                  line_width=2, annotation_text=f"PISO: R$ {PISO_LIQUIDEZ:,.0f}",
                  annotation_font_color="red", annotation_font_size=12)
    
    fig.update_layout(
        title="Simulação Estocástica — 3 Cenários de Viabilidade",
        xaxis_title="Período", yaxis_title="Saldo Projetado (R$)",
        template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


def fig_comparacao_modelos(sarimax_df, xgb_df, ensemble_df) -> go.Figure:
    """Compara as previsões dos 3 modelos lado a lado."""
    fig = go.Figure()
    
    if not sarimax_df.empty and 'SARIMAX_Pred' in sarimax_df.columns:
        fig.add_trace(go.Scatter(
            x=sarimax_df.index, y=sarimax_df['SARIMAX_Pred'],
            mode='lines+markers', name='SARIMAX',
            line=dict(color='#1f77b4', width=2)
        ))
    
    if not xgb_df.empty and 'XGBoost_Pred' in xgb_df.columns:
        fig.add_trace(go.Scatter(
            x=xgb_df.index, y=xgb_df['XGBoost_Pred'],
            mode='lines+markers', name='XGBoost',
            line=dict(color='#2ca02c', width=2)
        ))
    
    if not ensemble_df.empty and 'Ensemble_Pred' in ensemble_df.columns:
        fig.add_trace(go.Scatter(
            x=ensemble_df.index, y=ensemble_df['Ensemble_Pred'],
            mode='lines+markers', name='Ensemble (Final)',
            line=dict(color='#d62728', width=3, dash='dash')
        ))
    
    fig.add_hline(y=PISO_LIQUIDEZ, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Comparação: SARIMAX vs XGBoost vs Ensemble",
        template="plotly_white", height=400
    )
    return fig
