import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from prophet import Prophet
import warnings
import os
import logging

# Desativar avisos pesados do terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsões de Seleção - IA & Séries Temporais", layout="wide")

@st.cache_data
def carregar_e_processar_dados():
    try:
        df = pd.read_csv('all_matches.csv')
    except FileNotFoundError:
        return None

    df['date'] = pd.to_datetime(df['date'])
    date = df['date'].dt.strftime('%Y-%m-%d')
    splitted_date = date.str.split('-')
    df['year'] = [int(x[0]) for x in splitted_date]
    df = df[df['year'] >= 1910]

    def get_annual_stats(df):
        home_stats = df[['date', 'year', 'home_team', 'home_score', 'away_score']].rename(
            columns={'home_team': 'team', 'home_score': 'goals_for', 'away_score': 'goals_against'}
        )
        home_stats['result'] = np.where(home_stats['goals_for'] > home_stats['goals_against'], 3,
                                np.where(home_stats['goals_for'] == home_stats['goals_against'], 1, 0))

        away_stats = df[['date', 'year', 'away_team', 'away_score', 'home_score']].rename(
            columns={'away_team': 'team', 'away_score': 'goals_for', 'home_score': 'goals_against'}
        )
        away_stats['result'] = np.where(away_stats['goals_for'] > away_stats['goals_against'], 3,
                                np.where(away_stats['goals_for'] == away_stats['goals_against'], 1, 0))

        all_stats = pd.concat([home_stats, away_stats])
        annual = all_stats.groupby(['team', 'year']).agg(
            games=('result', 'count'),
            points=('result', 'sum'),
            goals_scored=('goals_for', 'mean'),
            goals_conceded=('goals_against', 'mean')
        ).reset_index()

        annual['performance_score'] = annual['points'] / annual['games']
        return annual

    df_annual = get_annual_stats(df)
    df_annual = df_annual.sort_values(['team', 'year'])
    df_annual['prev_score_1'] = df_annual.groupby('team')['performance_score'].shift(1)
    df_annual['prev_score_2'] = df_annual.groupby('team')['performance_score'].shift(2)
    df_annual['prev_goals_1'] = df_annual.groupby('team')['goals_scored'].shift(1)
    df_annual['prev_goals_2'] = df_annual.groupby('team')['goals_scored'].shift(2)
    return df_annual

st.markdown("""<style>div[data-baseweb="tooltip"] {font-size: 13px !important;}</style>""", unsafe_allow_html=True)

# --- INTERFACE: BARRA LATERAL ---
st.sidebar.title("Configuração da Simulação")

df_full = carregar_e_processar_dados()
if df_full is None:
    st.error("ERRO: O arquivo 'all_matches.csv' não foi encontrado na pasta.")
    st.stop()

# 1. SELETOR DE TÉCNICAS
lista_modelos_disponiveis = [
    "Random Forest (Machine Learning)", 
    "XGBoost (Machine Learning)", 
    "LSTM (Deep Learning)",
    "ARIMA (Série Temporal)",
    "Prophet (Série Temporal)",
    "Suavização Exponencial Dupla (Série Temporal)"
]

modelos_escolhidos = st.sidebar.multiselect(
    "Escolha as Técnicas para Comparar:", 
    options=lista_modelos_disponiveis,
    default=["Random Forest (Machine Learning)", "XGBoost (Machine Learning)"] 
)

anos_disponiveis = sorted(df_full['year'].unique())
target_year = st.sidebar.selectbox("Escolha o Ano de Previsão (Teste):", options=anos_disponiveis, index=len(anos_disponiveis)-1)
cutoff_year = target_year - 1

st.sidebar.markdown(f"""
---
**Resumo:**
* **Treinamento:** Até {cutoff_year}
* **Previsão:** {target_year}
""")

st.title(f"Guia de Seleções: Previsões de {target_year}")

if not modelos_escolhidos:
    st.warning("👈 Por favor, selecione pelo menos um modelo na barra lateral para começar.")
    st.stop()

# 1. FUNÇÃO DE TREINAMENTO COM CACHE
@st.cache_data(show_spinner=False)
def realizar_treinamento(df, ano_alvo, modelo):
    ano_corte = ano_alvo - 1
    resultados = []
    
    if "Random Forest" in modelo:
        df_rf = df.dropna().copy()
        features = ['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']
        X_train, y_train = df_rf[df_rf['year'] <= ano_corte][features], df_rf[df_rf['year'] <= ano_corte]['performance_score']
        X_test = df_rf[df_rf['year'] == ano_alvo][features]
        if not X_train.empty and not X_test.empty:
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            df_teste = df_rf[df_rf['year'] == ano_alvo].copy()
            df_teste['predicted_score'] = rf.predict(X_test)
            return df_teste
            
    elif "XGBoost" in modelo:
        df_xgb = df.dropna().copy()
        features = ['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']
        X_train, y_train = df_xgb[df_xgb['year'] <= ano_corte][features], df_xgb[df_xgb['year'] <= ano_corte]['performance_score']
        X_test = df_xgb[df_xgb['year'] == ano_alvo][features]
        if not X_train.empty and not X_test.empty:
            xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            df_teste = df_xgb[df_xgb['year'] == ano_alvo].copy()
            df_teste['predicted_score'] = xgb_model.predict(X_test).clip(0, 3)
            return df_teste

    elif "LSTM" in modelo:
        df_lstm = df.dropna().copy()
        features = ['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']
        X_train, y_train = df_lstm[df_lstm['year'] <= ano_corte][features], df_lstm[df_lstm['year'] <= ano_corte]['performance_score']
        X_test = df_lstm[df_lstm['year'] == ano_alvo][features]
        if not X_train.empty and not X_test.empty:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            model_lstm = Sequential()
            model_lstm.add(LSTM(32, activation='relu', input_shape=(1, len(features))))
            model_lstm.add(Dense(1))
            model_lstm.compile(optimizer='adam', loss='mse')
            model_lstm.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, verbose=0)
            df_teste = df_lstm[df_lstm['year'] == ano_alvo].copy()
            df_teste['predicted_score'] = np.clip(model_lstm.predict(X_test_reshaped, verbose=0).flatten(), 0, 3)
            return df_teste

    elif "ARIMA" in modelo:
        for time in df['team'].unique():
            dados_time = df[df['team'] == time].sort_values('year')
            dados_treino = dados_time[dados_time['year'] <= ano_corte]
            linha_atual = dados_time[dados_time['year'] == ano_alvo]
            if len(dados_treino) >= 5 and not linha_atual.empty:
                try:
                    model_fit = ARIMA(dados_treino['performance_score'].values, order=(1, 1, 1)).fit()
                    previsao = max(0, min(3, model_fit.forecast(steps=1)[0])) 
                    resultados.append({'team': time, 'year': ano_alvo, 'performance_score': linha_atual['performance_score'].values[0], 'predicted_score': previsao, 'games': linha_atual['games'].values[0], 'prev_score_1': dados_treino.iloc[-1]['performance_score'], 'prev_score_2': dados_treino.iloc[-2]['performance_score'] if len(dados_treino) > 1 else np.nan, 'prev_goals_1': dados_treino.iloc[-1]['goals_scored'], 'prev_goals_2': dados_treino.iloc[-2]['goals_scored'] if len(dados_treino) > 1 else np.nan})
                except Exception: pass
        if resultados: return pd.DataFrame(resultados)

    elif "Prophet" in modelo:
        for time in df['team'].unique():
            dados_time = df[df['team'] == time].sort_values('year')
            dados_treino, linha_atual = dados_time[dados_time['year'] <= ano_corte], dados_time[dados_time['year'] == ano_alvo]
            if len(dados_treino) >= 5 and not linha_atual.empty:
                try:
                    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False).fit(pd.DataFrame({'ds': pd.to_datetime(dados_treino['year'], format='%Y'), 'y': dados_treino['performance_score']}))
                    previsao = max(0, min(3, m.predict(pd.DataFrame({'ds': [pd.to_datetime(ano_alvo, format='%Y')]}))['yhat'].values[0])) 
                    resultados.append({'team': time, 'year': ano_alvo, 'performance_score': linha_atual['performance_score'].values[0], 'predicted_score': previsao, 'games': linha_atual['games'].values[0], 'prev_score_1': dados_treino.iloc[-1]['performance_score'], 'prev_score_2': dados_treino.iloc[-2]['performance_score'] if len(dados_treino) > 1 else np.nan, 'prev_goals_1': dados_treino.iloc[-1]['goals_scored'], 'prev_goals_2': dados_treino.iloc[-2]['goals_scored'] if len(dados_treino) > 1 else np.nan})
                except Exception: pass
        if resultados: return pd.DataFrame(resultados)

    elif "Exponencial" in modelo:
        for time in df['team'].unique():
            dados_time = df[df['team'] == time].sort_values('year')
            dados_treino, linha_atual = dados_time[dados_time['year'] <= ano_corte], dados_time[dados_time['year'] == ano_alvo]
            if len(dados_treino) >= 5 and not linha_atual.empty:
                try:
                    model_fit = ExponentialSmoothing(dados_treino['performance_score'].values, trend='add', seasonal=None, initialization_method="estimated").fit()
                    previsao = max(0, min(3, model_fit.forecast(1)[0])) 
                    resultados.append({'team': time, 'year': ano_alvo, 'performance_score': linha_atual['performance_score'].values[0], 'predicted_score': previsao, 'games': linha_atual['games'].values[0], 'prev_score_1': dados_treino.iloc[-1]['performance_score'], 'prev_score_2': dados_treino.iloc[-2]['performance_score'] if len(dados_treino) > 1 else np.nan, 'prev_goals_1': dados_treino.iloc[-1]['goals_scored'], 'prev_goals_2': dados_treino.iloc[-2]['goals_scored'] if len(dados_treino) > 1 else np.nan})
                except Exception: pass
        if resultados: return pd.DataFrame(resultados)
    return None

# 2. EXECUÇÃO MULTIPLA
dict_resultados = {}
for mod in modelos_escolhidos:
    with st.spinner(f'Treinando {mod.split(" ")[0]}...'):
        df_res = realizar_treinamento(df_full, target_year, mod)
        if df_res is not None and not df_res.empty:
            dict_resultados[mod] = df_res

if not dict_resultados:
    st.error("Nenhum modelo conseguiu gerar previsões com os dados atuais.")
    st.stop()

# --- DASHBOARD VISUAL ---

# 1. TABELA GLOBAL (LEADERBOARD)
st.divider()
st.subheader("Qual técnica previu melhor?")

dados_tabela = []
for mod, df_m in dict_resultados.items():
    mae = mean_absolute_error(df_m['performance_score'], df_m['predicted_score'])
    rmse = np.sqrt(mean_squared_error(df_m['performance_score'], df_m['predicted_score']))
    corr = df_m['performance_score'].corr(df_m['predicted_score']) if df_m['predicted_score'].nunique() > 1 else 0.0
    dados_tabela.append({"Técnica": mod.split(' ')[0], "Erro Médio (MAE) ↓": mae, "RMSE ↓": rmse, "Correlação (R) ↑": corr})

df_leaderboard = pd.DataFrame(dados_tabela).set_index("Técnica")
# Destaca os melhores resultados na tabela
st.dataframe(df_leaderboard.style.highlight_min(subset=["Erro Médio (MAE) ↓", "RMSE ↓"], color='#90ee90')
                               .highlight_max(subset=["Correlação (R) ↑"], color='#90ee90'), use_container_width=True)


# 2. SELETOR DE TIME
st.markdown("---")
primeiro_modelo = list(dict_resultados.keys())[0]
df_referencia = dict_resultados[primeiro_modelo]
times_no_ano = list(sorted(df_referencia['team'].unique()))

if 'time_selecionado' not in st.session_state or st.session_state['time_selecionado'] not in times_no_ano:
    st.session_state['time_selecionado'] = times_no_ano[0]

indice_atual = times_no_ano.index(st.session_state['time_selecionado'])

col_sel_1, col_sel_2 = st.columns([1, 3])
with col_sel_1:
    novo_time = st.selectbox("Analisar Seleção Específica:", times_no_ano, index=indice_atual)
    if novo_time != st.session_state['time_selecionado']:
        st.session_state['time_selecionado'] = novo_time
        st.rerun()

time_selecionado = st.session_state['time_selecionado']
real_score = df_referencia[df_referencia['team'] == time_selecionado].iloc[0]['performance_score']

st.subheader(f"Comparativo Específico: {time_selecionado} (Real: {real_score:.2f})")

tab1, tab2 = st.tabs(["Comparativo Entre Técnicas", "Dispersão Global"])

with tab1:
    fig_simples = go.Figure()
    
    # Adiciona a barra do Real no eixo X 
    fig_simples.add_trace(go.Bar(
        x=['<b>Performance<br>Real</b>'], y=[real_score], 
        marker_color='red', text=[f"{real_score:.2f}"], textposition='inside', textfont=dict(size=16, color='white'),
        hovertemplate="Real: %{y:.2f}<extra></extra>"
    ))
    
    cores_modelos = px.colors.qualitative.Pastel
    
    # Adiciona as colunas de cada modelo selecionado
    for i, mod in enumerate(modelos_escolhidos):
        if time_selecionado in dict_resultados[mod]['team'].values:
            pred = dict_resultados[mod][dict_resultados[mod]['team'] == time_selecionado].iloc[0]['predicted_score']
            nome_curto = mod.split(' ')[0]
            fig_simples.add_trace(go.Bar(
                x=[f"Previsão<br>({nome_curto})"], y=[pred], 
                marker_color=cores_modelos[i % len(cores_modelos)], text=[f"{pred:.2f}"], textposition='inside', textfont=dict(size=14, color='black'),
                hovertemplate=f"{nome_curto}: %{{y:.2f}}<extra></extra>"
            ))

    fig_simples.update_layout(
        height=400,
        yaxis=dict(range=[0, 3.2], showticklabels=False, title=""), 
        xaxis=dict(title="", tickangle=0), 
        margin=dict(l=20, r=20, t=30, b=40),
        showlegend=False,
        barmode='group'
    )
    st.plotly_chart(fig_simples, use_container_width=True)

    st.markdown("Dados Históricos Utilizados")
    
    df_team = df_referencia[df_referencia['team'] == time_selecionado].iloc[0]

    features_data = {
        'Métrica': [
            'Performance No Ano Anterior', 
            'Performance de 2 Anos Atrás', 
            'Média De Gols Marcados No Ano Anterior',
            'Média De Gols Marcados 2 Anos Atrás'  
        ],
        'Valor': [
            f"{df_team['prev_score_1']:.4f}", 
            f"{df_team['prev_score_2']:.4f}", 
            f"{df_team['prev_goals_1']:.2f}",
            f"{df_team['prev_goals_2']:.2f}" 
        ]
    }
    st.table(pd.DataFrame(features_data).set_index('Métrica'))

with tab2:
    st.caption("Clique em qualquer bolinha para analisar aquela seleção!")
    fig_scatter = go.Figure()
    
    cores_linhas = px.colors.qualitative.Plotly
    
    # Adiciona os pontos para cada modelo selecionado
    for i, mod in enumerate(modelos_escolhidos):
        df_m = dict_resultados[mod]
        nome_curto = mod.split(' ')[0]
        
        df_others = df_m[df_m['team'] != time_selecionado]
        fig_scatter.add_trace(go.Scatter(
            x=df_others['performance_score'], y=df_others['predicted_score'], mode='markers', 
            name=nome_curto, marker=dict(color=cores_linhas[i % len(cores_linhas)], size=6, opacity=0.4), 
            text=df_others['team'], customdata=df_others['team'], legendgroup=nome_curto
        ))
        
        df_sel = df_m[df_m['team'] == time_selecionado]
        if not df_sel.empty:
            fig_scatter.add_trace(go.Scatter(
                x=df_sel['performance_score'], y=df_sel['predicted_score'], mode='markers', 
                name=f"{nome_curto} ({time_selecionado})", 
                marker=dict(color=cores_linhas[i % len(cores_linhas)], size=16, symbol='star', line=dict(width=2, color='black')), 
                text=df_sel['team'], customdata=df_sel['team'], legendgroup=nome_curto
            ))

    fig_scatter.add_shape(type="line", line=dict(dash='dash', color='gray'), x0=0, y0=0, x1=3, y1=3)
    fig_scatter.update_layout(xaxis_title="Performance Real", yaxis_title="Performance Prevista", height=500, clickmode='event+select')
    
    evento_clique = st.plotly_chart(fig_scatter, use_container_width=True, on_select="rerun", selection_mode="points")
    
    if evento_clique and len(evento_clique.selection['points']) > 0:
        ponto = evento_clique.selection['points'][0]
        time_clicado = ponto.get('text', None)
        if time_clicado is None and 'customdata' in ponto:
            time_clicado = ponto['customdata'][0] if isinstance(ponto['customdata'], list) else ponto['customdata']
        time_clicado = str(time_clicado).strip("[]'\" ")
        
        if time_clicado in times_no_ano and time_clicado != st.session_state['time_selecionado']:
            st.session_state['time_selecionado'] = time_clicado
            st.rerun()
