import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
from xgboost import XGBRegressor

# Ignorar os avisos de convergência do ARIMA
warnings.filterwarnings("ignore")

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsões de Seleção - IA & Séries Temporais", layout="wide")

# --- FUNÇÕES DE PROCESSAMENTO (CACHED) ---
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

    # Lags para Random Forest
    df_annual = df_annual.sort_values(['team', 'year'])
    df_annual['prev_score_1'] = df_annual.groupby('team')['performance_score'].shift(1)
    df_annual['prev_score_2'] = df_annual.groupby('team')['performance_score'].shift(2)
    df_annual['prev_goals_1'] = df_annual.groupby('team')['goals_scored'].shift(1)
    df_annual['prev_goals_2'] = df_annual.groupby('team')['goals_scored'].shift(2)
    
    return df_annual

st.markdown("""
<style>
    div[data-baseweb="tooltip"] {
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- INTERFACE: BARRA LATERAL ---
st.sidebar.title("Configuração da Simulação")

df_full = carregar_e_processar_dados()
if df_full is None:
    st.error("ERRO: O arquivo 'all_matches.csv' não foi encontrado na pasta.")
    st.stop()

# 1. Seletor de Modelo
modelo_escolhido = st.sidebar.selectbox(
    "Escolha a Técnica de Previsão:", 
    options=["Random Forest (Machine Learning)", "XGBoost (Machine Learning)", "ARIMA (Série Temporal)"]
)

# 2. Seletor de Ano
anos_disponiveis = sorted(df_full['year'].unique())
target_year = st.sidebar.selectbox(
    "Escolha o Ano de Previsão (Teste):", 
    options=anos_disponiveis, 
    index=len(anos_disponiveis)-1
)
cutoff_year = target_year - 1

st.sidebar.markdown(f"""
---
**Resumo do Treino:**
* **Técnica:** {modelo_escolhido.split(' ')[0]}
* **Treinamento:** Até {cutoff_year}
* **Previsão:** {target_year}
""")

st.title(f"Previsão de Performance: {target_year}")
st.markdown(f"**Técnica em uso:** {modelo_escolhido}")

# 1. CRIAMOS A FUNÇÃO DE TREINAMENTO COM CACHE
@st.cache_data(show_spinner=False)
def realizar_treinamento(df, ano_alvo, modelo):
    ano_corte = ano_alvo - 1
    resultados = []
    
    if modelo == "Random Forest (Machine Learning)":
        df_rf = df.dropna().copy()
        features = ['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']
        target = 'performance_score'

        X_train = df_rf[df_rf['year'] <= ano_corte][features]
        y_train = df_rf[df_rf['year'] <= ano_corte][target]
        X_test = df_rf[df_rf['year'] == ano_alvo][features]
        
        if not X_train.empty and not X_test.empty:
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            df_teste = df_rf[df_rf['year'] == ano_alvo].copy()
            df_teste['predicted_score'] = rf.predict(X_test)
            df_teste['abs_error'] = abs(df_teste['performance_score'] - df_teste['predicted_score'])
            return df_teste
        return None
    
    elif modelo == "XGBoost (Machine Learning)":
        df_xgb = df.dropna().copy()
        features = ['prev_score_1', 'prev_score_2', 'prev_goals_1', 'prev_goals_2']
        target = 'performance_score'

        X_train = df_xgb[df_xgb['year'] <= ano_corte][features]
        y_train = df_xgb[df_xgb['year'] <= ano_corte][target]
        X_test = df_xgb[df_xgb['year'] == ano_alvo][features]
        
        if not X_train.empty and not X_test.empty:
            # Instanciando o XGBoost (Parâmetros padrão eficientes)
            xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            
            df_teste = df_xgb[df_xgb['year'] == ano_alvo].copy()
            df_teste['predicted_score'] = xgb_model.predict(X_test)
            
            # Limita a previsão aos limites lógicos (0 a 3 pontos)
            df_teste['predicted_score'] = df_teste['predicted_score'].clip(lower=0, upper=3)
            
            df_teste['abs_error'] = abs(df_teste['performance_score'] - df_teste['predicted_score'])
            return df_teste
        return None

    elif modelo == "ARIMA (Série Temporal)":
        times = df['team'].unique()
        for time in times:
            dados_time = df[df['team'] == time].sort_values('year')
            dados_treino = dados_time[dados_time['year'] <= ano_corte]
            linha_atual = dados_time[dados_time['year'] == ano_alvo]

            if len(dados_treino) >= 5 and not linha_atual.empty:
                y_train = dados_treino['performance_score'].values
                valor_real = linha_atual['performance_score'].values[0]
                jogos = linha_atual['games'].values[0]
                
                try:
                    model_arima = ARIMA(y_train, order=(1, 1, 1))
                    model_fit = model_arima.fit()
                    
                    previsao = max(0, min(3, model_fit.forecast(steps=1)[0])) # Limita entre 0 e 3

                    resultados.append({
                            'team': time,
                            'year': ano_alvo,
                            'performance_score': valor_real,
                            'predicted_score': previsao,
                            'games': jogos,
                            'prev_score_1': dados_treino.iloc[-1]['performance_score'],
                            'prev_score_2': dados_treino.iloc[-2]['performance_score'] if len(dados_treino) > 1 else np.nan,
                            'prev_goals_1': dados_treino.iloc[-1]['goals_scored'],
                            # --- ADICIONE ESTA LINHA AQUI ---
                            'prev_goals_2': dados_treino.iloc[-2]['goals_scored'] if len(dados_treino) > 1 else np.nan
                        })
                except Exception:
                    pass
        
        if resultados:
            df_arima = pd.DataFrame(resultados)
            df_arima['abs_error'] = abs(df_arima['performance_score'] - df_arima['predicted_score'])
            return df_arima
        return None

# 2. EXECUTAMOS A FUNÇÃO
# O Spinner roda apenas se a função precisar calcular. Se estiver no cache, pula direto.
with st.spinner(f'Calculando previsões usando {modelo_escolhido.split(" ")[0]}...'):
    df_res = realizar_treinamento(df_full, target_year, modelo_escolhido)

# 3. VERIFICAÇÃO DE ERROS
if df_res is None or df_res.empty:
    st.error(f"Dados históricos insuficientes para treinar o modelo {modelo_escolhido} no ano de {target_year}.")
    st.stop()
# --- DASHBOARD VISUAL ---

# 1. KPIs Globais
st.divider()
col1, col2, col3, col4 = st.columns(4)

mae = mean_absolute_error(df_res['performance_score'], df_res['predicted_score'])
rmse = np.sqrt(mean_squared_error(df_res['performance_score'], df_res['predicted_score']))

# Proteção para correlação caso o ARIMA preveja uma linha reta (variância zero)
if df_res['predicted_score'].nunique() > 1:
    correlacao = df_res['performance_score'].corr(df_res['predicted_score'])
else:
    correlacao = 0.0

col1.metric("Erro Médio (MAE)", f"{mae:.4f}", help="Média da diferença absoluta entre o Real e o Previsto.")
col2.metric("RMSE", f"{rmse:.4f}", help="Penaliza erros grandes de forma mais severa.")
col3.metric("Correlação (R)", f"{correlacao:.4f}", help="Varia de -1 a 1. Mede a capacidade do modelo de acertar o ranking das forças das seleções.")
col4.metric("Seleções Analisadas", len(df_res), help="O ARIMA filtra seleções com menos de 5 anos de histórico.")

# 2. Seletor de Time
st.markdown("---")
times_no_ano = sorted(df_res['team'].unique())
col_sel_1, col_sel_2 = st.columns([1, 3])
with col_sel_1:
    time_selecionado = st.selectbox("Analisar Seleção Específica:", times_no_ano)

df_team = df_res[df_res['team'] == time_selecionado].iloc[0]

# 3. Métricas do Time
st.subheader(f"Detalhes: {time_selecionado}")
c1, c2, c3 = st.columns(3)
real = df_team['performance_score']
pred = df_team['predicted_score']

diferenca = pred - real 

c1.metric("Real", f"{real:.2f}")

c2.metric(
    f"Previsto ({modelo_escolhido.split(' ')[0]})", 
    f"{pred:.2f}", 
    delta=f"{diferenca:.2f}", 
    delta_color="normal",
    help="🟢 Seta Verde: Superestimou. \n🔴 Seta Vermelha: Subestimou."
)
c3.metric("Jogos no Ano", int(df_team['games']))

# 4. Gráficos e Tabelas
tab1, tab2 = st.tabs(["Comparativo Individual", "Dispersão Global"])

with tab1:
    fig_simples = go.Figure()
    
    fig_simples.add_trace(go.Bar(
        y=['<b>Previsão</b>', '<b>Real</b>'], 
        x=[pred, real], 
        orientation='h',
        marker_color=['#1f77b4', '#2ca02c'], # Azul para previsto, Verde para real
        text=[f"{pred:.2f}", f"{real:.2f}"], 
        textposition='inside',
        textfont=dict(size=16, color='white')
    ))
    
    fig_simples.update_layout(
        height=250, # Altura reduzida para ficar mais "clean"
        xaxis=dict(range=[0, 3.2], title="Pontuação (0 a 3)"),
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    
    st.plotly_chart(fig_simples, use_container_width=True)
    
    st.markdown("##### Dados Anteriores")

    features_data = {
        'Métrica': [
            'Performance No Ano Anterior', 
            'Performance de 2 Anos Atrás', 
            'Média De Gols Marcados No Ano Anterior',
            'Média De Gols Marcados 2 Anos Atrás'  # <--- NOVA LINHA
        ],
        'Valor': [
            f"{df_team['prev_score_1']:.4f}", 
            f"{df_team['prev_score_2']:.4f}", 
            f"{df_team['prev_goals_1']:.2f}",
            f"{df_team['prev_goals_2']:.2f}"       # <--- NOVA LINHA
        ]
    }
    st.table(pd.DataFrame(features_data).set_index('Métrica'))

with tab2:
    st.markdown("##### Comparação de todas as seleções (Global)")
    
    colors = ['red' if t == time_selecionado else 'blue' for t in df_res['team']]
    
    fig_scatter = go.Figure()
    
    df_others = df_res[df_res['team'] != time_selecionado]
    fig_scatter.add_trace(go.Scatter(x=df_others['performance_score'], y=df_others['predicted_score'],
        mode='markers', name='Outras Seleções', marker=dict(color='#A6C8FF', size=8, opacity=0.6), text=df_others['team']))
    
    fig_scatter.add_trace(go.Scatter(x=[real], y=[pred], mode='markers', name=time_selecionado,
        marker=dict(color='red', size=12, line=dict(width=2, color='black')), text=[time_selecionado]))

    fig_scatter.add_shape(type="line", line=dict(dash='dash', color='gray'), x0=0, y0=0, x1=3, y1=3)
    fig_scatter.update_layout(xaxis_title="Performance Real", yaxis_title="Performance Prevista", height=500)
    
    st.plotly_chart(fig_scatter, use_container_width=True)
