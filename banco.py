import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
import os
import logging
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

print("Carregando e processando os dados históricos...")
try:
    df = pd.read_csv('all_matches.csv')
except FileNotFoundError:
    print("ERRO: 'all_matches.csv' não encontrado.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[df['year'] >= 1910]

home_stats = df[['year', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'home_score': 'goals_for', 'away_score': 'goals_against'})
home_stats['result'] = np.where(home_stats['goals_for'] > home_stats['goals_against'], 3, np.where(home_stats['goals_for'] == home_stats['goals_against'], 1, 0))
away_stats = df[['year', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'away_score': 'goals_for', 'home_score': 'goals_against'})
away_stats['result'] = np.where(away_stats['goals_for'] > away_stats['goals_against'], 3, np.where(away_stats['goals_for'] == away_stats['goals_against'], 1, 0))

all_stats = pd.concat([home_stats, away_stats])
df_annual = all_stats.groupby(['team', 'year']).agg(games=('result', 'count'), points=('result', 'sum'), goals_scored=('goals_for', 'mean'), goals_conceded=('goals_against', 'mean')).reset_index()
df_annual['performance_score'] = df_annual['points'] / df_annual['games']
df_annual = df_annual.sort_values(['team', 'year'])
df_annual['prev_score_1'] = df_annual.groupby('team')['performance_score'].shift(1)
df_annual['prev_score_2'] = df_annual.groupby('team')['performance_score'].shift(2)
df_annual['prev_goals_1'] = df_annual.groupby('team')['goals_scored'].shift(1)
df_annual['prev_goals_2'] = df_annual.groupby('team')['goals_scored'].shift(2)
df_annual = df_annual.dropna()

# --- CONFIGURAÇÕES DA GERAÇÃO ---
anos_de_corte = range(1910, 2025) # Vai treinar os modelos parando em 1910, depois em 1911, etc.
horizontes_maximos = 10            # Vai prever até 10 anos pra frente de cada corte
modelos = ["Random Forest", "XGBoost", "LSTM", "ARIMA", "Prophet", "Exponencial"]
arquivo_saida = 'previsoes_master.csv'

print(f"🚀 Iniciando o treinamento em massa. Isso pode demorar horas. Deixe rodando...\n")

# Se o arquivo já existir, cria um novo em branco com as colunas
pd.DataFrame(columns=['cutoff_year', 'predicted_year', 'model', 'team', 'predicted_score']).to_csv(arquivo_saida, index=False)

for ano_corte in anos_de_corte:
    print(f"\n--- Treinando modelos com dados ATÉ {ano_corte} ---")
    resultados_batch = []
    anos_alvo = list(range(ano_corte + 1, ano_corte + horizontes_maximos + 1))
    
    for mod in modelos:
        print(f"[{ano_corte}] Treinando {mod}...")
        try:
            if mod == "Random Forest":
                X_train, y_train = df_annual[df_annual['year'] <= ano_corte][['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']], df_annual[df_annual['year'] <= ano_corte]['performance_score']
                df_teste = df_annual[df_annual['year'].isin(anos_alvo)].copy()
                if not X_train.empty and not df_teste.empty:
                    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
                    df_teste['predicted_score'] = rf.predict(df_teste[['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']])
                    for _, row in df_teste.iterrows(): resultados_batch.append({'cutoff_year': ano_corte, 'predicted_year': row['year'], 'model': mod, 'team': row['team'], 'predicted_score': row['predicted_score']})

            elif mod == "XGBoost":
                X_train, y_train = df_annual[df_annual['year'] <= ano_corte][['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']], df_annual[df_annual['year'] <= ano_corte]['performance_score']
                df_teste = df_annual[df_annual['year'].isin(anos_alvo)].copy()
                if not X_train.empty and not df_teste.empty:
                    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train)
                    df_teste['predicted_score'] = xgb.predict(df_teste[['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']]).clip(0, 3)
                    for _, row in df_teste.iterrows(): resultados_batch.append({'cutoff_year': ano_corte, 'predicted_year': row['year'], 'model': mod, 'team': row['team'], 'predicted_score': row['predicted_score']})

            elif mod == "LSTM":
                X_train, y_train = df_annual[df_annual['year'] <= ano_corte][['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']], df_annual[df_annual['year'] <= ano_corte]['performance_score']
                df_teste = df_annual[df_annual['year'].isin(anos_alvo)].copy()
                if not X_train.empty and not df_teste.empty:
                    scaler = StandardScaler()
                    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(df_teste[['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']])
                    X_train_reshaped, X_test_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, 4)), X_test_scaled.reshape((X_test_scaled.shape[0], 1, 4))
                    lstm = Sequential([LSTM(32, activation='relu', input_shape=(1, 4)), Dense(1)])
                    lstm.compile(optimizer='adam', loss='mse')
                    lstm.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, verbose=0)
                    df_teste['predicted_score'] = np.clip(lstm.predict(X_test_reshaped, verbose=0).flatten(), 0, 3)
                    for _, row in df_teste.iterrows(): resultados_batch.append({'cutoff_year': ano_corte, 'predicted_year': row['year'], 'model': mod, 'team': row['team'], 'predicted_score': row['predicted_score']})

            elif mod in ["ARIMA", "Prophet", "Exponencial"]:
                for time in df_annual['team'].unique():
                    dados_treino = df_annual[(df_annual['team'] == time) & (df_annual['year'] <= ano_corte)]
                    if len(dados_treino) >= 5:
                        ultimo_ano = dados_treino['year'].iloc[-1]
                        passos = int(max(anos_alvo) - ultimo_ano)
                        if passos > 0:
                            try:
                                if mod == "ARIMA":
                                    prevs = ARIMA(dados_treino['performance_score'].values, order=(1, 1, 1)).fit().forecast(steps=passos)
                                    for i, p in enumerate(prevs):
                                        ano_f = ultimo_ano + i + 1
                                        if ano_f in anos_alvo: resultados_batch.append({'cutoff_year': ano_corte, 'predicted_year': ano_f, 'model': mod, 'team': time, 'predicted_score': max(0, min(3, p))})
                                elif mod == "Prophet":
                                    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False).fit(pd.DataFrame({'ds': pd.to_datetime(dados_treino['year'], format='%Y'), 'y': dados_treino['performance_score']}))
                                    forecast = m.predict(pd.DataFrame({'ds': pd.to_datetime([a for a in anos_alvo if a > ultimo_ano], format='%Y')}))
                                    for _, row in forecast.iterrows():
                                        if row['ds'].year in anos_alvo: resultados_batch.append({'cutoff_year': ano_corte, 'predicted_year': row['ds'].year, 'model': mod, 'team': time, 'predicted_score': max(0, min(3, row['yhat']))})
                                elif mod == "Exponencial":
                                    prevs = ExponentialSmoothing(dados_treino['performance_score'].values, trend='add', seasonal=None, initialization_method="estimated").fit().forecast(passos)
                                    for i, p in enumerate(prevs):
                                        ano_f = ultimo_ano + i + 1
                                        if ano_f in anos_alvo: resultados_batch.append({'cutoff_year': ano_corte, 'predicted_year': ano_f, 'model': mod, 'team': time, 'predicted_score': max(0, min(3, p))})
                            except Exception: pass
        except Exception as e: print(f"Erro no modelo {mod}: {e}")
    
    # Salva os resultados do ano de corte atual no CSV para não perder tudo se houver erro ou queda de energia
    pd.DataFrame(resultados_batch).to_csv(arquivo_saida, mode='a', header=False, index=False)
    print(f" Dados até {ano_corte} salvos com sucesso!")

print("\nBANCO DE DADOS GERADO COM SUCESSO! Você já pode abrir o seu aplicativo Streamlit.")