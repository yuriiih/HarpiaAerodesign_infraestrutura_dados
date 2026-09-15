# src/models.py
"""
Modelos Preditivos: SARIMAX (séries temporais) + XGBoost (ensemble)
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

PISO_LIQUIDEZ = 5000.0
HORIZONTE_MESES = 6


def train_sarimax(df_gold: pd.DataFrame, horizon: int = HORIZONTE_MESES) -> pd.DataFrame:
    """
    SARIMAX com covariáveis exógenas.
    Fallback para ARIMA simples se dados insuficientes.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        logger.warning("statsmodels não instalado. Pulando SARIMAX.")
        return pd.DataFrame()
    
    serie = df_gold['Saldo_Acumulado'].copy()
    
    # SARIMAX precisa de pelo menos ~12 observações para sazonalidade
    use_sazonal = len(serie) >= 12
    
    try:
        if use_sazonal:
            model = SARIMAX(
                serie,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            model = SARIMAX(
                serie,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        
        results = model.fit(disp=False, maxiter=200)
        
        # Forecast
        forecast = results.get_forecast(steps=horizon)
        pred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        
        datas_futuras = pd.date_range(
            start=serie.index[-1] + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )
        
        resultado = pd.DataFrame({
            'Data': datas_futuras,
            'SARIMAX_Pred': pred.values,
            'SARIMAX_Lower': conf_int.iloc[:, 0].values,
            'SARIMAX_Upper': conf_int.iloc[:, 1].values,
        }).set_index('Data')
        
        logger.info("SARIMAX treinado com sucesso")
        return resultado
        
    except Exception as e:
        logger.error(f"Erro no SARIMAX: {e}")
        return pd.DataFrame()


def train_xgboost(df_gold: pd.DataFrame, horizon: int = HORIZONTE_MESES) -> pd.DataFrame:
    """
    XGBoost Regressor com features temporais.
    Captura não-linearidades e picos abruptos.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.warning("xgboost não instalado. Pulando XGBoost.")
        return pd.DataFrame()
    
    if len(df_gold) < 4:
        logger.warning("Dados insuficientes para XGBoost (mínimo 4 meses)")
        return pd.DataFrame()
    
    # Feature engineering
    df = df_gold.copy()
    df['mes_num'] = df.index.month
    df['mes_num_sq'] = df['mes_num'] ** 2
    df['receita_lag1'] = df['Receita'].shift(1)
    df['despesa_lag1'] = df['Despesa'].shift(1)
    df['saldo_lag1'] = df['Saldo_Acumulado'].shift(1)
    df['saldo_lag2'] = df['Saldo_Acumulado'].shift(2)
    df = df.dropna()
    
    features = ['mes_num', 'mes_num_sq', 'receita_lag1', 'despesa_lag1', 'saldo_lag1', 'saldo_lag2']
    
    X = df[features]
    y = df['Saldo_Acumulado']
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    model.fit(X, y)
    
    # Previsão iterativa (multi-step)
    last_row = df_gold.iloc[-1]
    previsoes = []
    datas_futuras = pd.date_range(
        start=df_gold.index[-1] + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'
    )
    
    saldo_atual = last_row['Saldo_Acumulado']
    receita_lag = last_row['Receita']
    despesa_lag = last_row['Despesa']
    saldo_lag1 = saldo_atual
    saldo_lag2 = df_gold['Saldo_Acumulado'].iloc[-2] if len(df_gold) > 1 else saldo_atual
    
    for data in datas_futuras:
        X_pred = pd.DataFrame([{
            'mes_num': data.month,
            'mes_num_sq': data.month ** 2,
            'receita_lag1': receita_lag,
            'despesa_lag1': despesa_lag,
            'saldo_lag1': saldo_lag1,
            'saldo_lag2': saldo_lag2,
        }])
        
        pred = model.predict(X_pred)[0]
        previsoes.append(pred)
        
        # Atualiza lags para próxima iteração
        saldo_lag2 = saldo_lag1
        saldo_lag1 = pred
        receita_lag = receita_lag * 0.95  # decaimento conservador
        despesa_lag = despesa_lag * 1.05  # inflação conservadora
    
    resultado = pd.DataFrame({
        'Data': datas_futuras,
        'XGBoost_Pred': previsoes,
    }).set_index('Data')
    
    logger.info("XGBoost treinado com sucesso")
    return resultado


def ensemble_forecast(sarimax_df: pd.DataFrame, xgb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina SARIMAX e XGBoost com média ponderada.
    SARIMAX tem peso maior quando há dados sazonais suficientes.
    """
    if sarimax_df.empty and xgb_df.empty:
        return pd.DataFrame()
    
    if sarimax_df.empty:
        return xgb_df.rename(columns={'XGBoost_Pred': 'Ensemble_Pred'})
    
    if xgb_df.empty:
        return sarimax_df.rename(columns={'SARIMAX_Pred': 'Ensemble_Pred'})
    
    # Alinha índices
    combined = sarimax_df[['SARIMAX_Pred']].join(xgb_df[['XGBoost_Pred']], how='inner')
    
    # Peso: 60% SARIMAX, 40% XGBoost
    combined['Ensemble_Pred'] = 0.6 * combined['SARIMAX_Pred'] + 0.4 * combined['XGBoost_Pred']
    
    return combined


def run_models(gold_data: dict) -> dict:
    """Orquestra todos os modelos."""
    df_monthly = gold_data['monthly']
    
    sarimax = train_sarimax(df_monthly)
    xgb = train_xgboost(df_monthly)
    ensemble = ensemble_forecast(sarimax, xgb)
    
    return {
        'sarimax': sarimax,
        'xgboost': xgb,
        'ensemble': ensemble,
    }
