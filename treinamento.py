import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsões de Seleção - IA & Séries Temporais", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    div[data-baseweb="tooltip"] { font-size: 13px !important; }
    span[data-baseweb="tag"] { background-color: #1f77b4 !important; color: white !important; }
    button[data-baseweb="tab"] * { color: black !important; }
    div[data-baseweb="tab"] * { color: black !important; }
    div[data-baseweb="tab-highlight"] { background-color: black !important; }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE CARREGAMENTO ---
@st.cache_data
def carregar_dados_brutos():
    try:
        return pd.read_csv('all_matches.csv')
    except FileNotFoundError:
        st.error("ERRO: 'all_matches.csv' não encontrado.")
        st.stop()

@st.cache_data
def carregar_dados_anuais():
    df = carregar_dados_brutos()
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
    return df_annual

@st.cache_data
def carregar_banco_previsoes():
    try:
        return pd.read_csv('previsoes_master.csv')
    except FileNotFoundError:
        st.error("ERRO: Banco de dados 'previsoes_master.csv' não encontrado. Rode o script gerar_banco.py primeiro.")
        st.stop()

df_bruto = carregar_dados_brutos()
df_full = carregar_dados_anuais()
df_previsoes_banco = carregar_banco_previsoes()


# --- MENU PRINCIPAL ---
st.sidebar.title("Navegação")
modo_app = st.sidebar.radio(
    "Escolha o Painel:",
    ["Visão Esportiva", "Visão Técnica"]
)

# MÓDULO 1: PAINEL DE FUTEBOL E DADOS
if modo_app == "Visão Esportiva":
    st.title("Seleções: Caminho para 2026")
    st.markdown("Bem-vindo ao painel esportivo! Utilizamos uma Inteligência Artificial para identificar as forças de cada seleção para o ano de 2026.")
    
    df_2026 = df_previsoes_banco[df_previsoes_banco['predicted_year'] == 2026]
    if not df_2026.empty:
        corte_recente = df_2026['cutoff_year'].max()
        df_2026 = df_2026[df_2026['cutoff_year'] == corte_recente]
        
        df_consenso = df_2026.groupby('team')['predicted_score'].mean().reset_index()
        df_consenso = df_consenso.sort_values('predicted_score', ascending=False)
        
        # NOVO: Adicionamos a terceira aba "Máquina do Tempo (Ano a Ano)"
        tab_ranking, tab_selecao, tab_ano = st.tabs(["Ranking Global de Favoritismo", "Análise Individual por Seleção", "Dados por ano"])
        
        with tab_ranking:
            st.markdown("##### As 15 Seleções mais fortes para 2026")
            top_15 = df_consenso.head(15)
            
            fig_bar = px.bar(
                top_15, x='predicted_score', y='team', orientation='h',
                labels={'predicted_score': 'Força Prevista (0 a 3)', 'team': 'Seleção'}
            )
            fig_bar.update_traces(marker_color='#1f77b4')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=500, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_bar, width='stretch')
            
        with tab_selecao:
            col1, col2 = st.columns([1, 3])
            with col1:
                times_disponiveis = sorted(df_consenso['team'].unique())
                idx_padrao = times_disponiveis.index("Brazil") if "Brazil" in times_disponiveis else 0
                time_escolhido = st.selectbox("Escolha uma Seleção:", times_disponiveis, index=idx_padrao)
            
            st.markdown(f"### Histórico Total: {time_escolhido}")
            
            df_jogos_time = df_bruto[(df_bruto['home_team'] == time_escolhido) | (df_bruto['away_team'] == time_escolhido)].copy()
            df_jogos_time['is_home'] = df_jogos_time['home_team'] == time_escolhido
            
            total_jogos = len(df_jogos_time)
            vitorias = len(df_jogos_time[(df_jogos_time['is_home'] & (df_jogos_time['home_score'] > df_jogos_time['away_score'])) | (~df_jogos_time['is_home'] & (df_jogos_time['away_score'] > df_jogos_time['home_score']))])
            taxa_vitoria = (vitorias / total_jogos) * 100 if total_jogos > 0 else 0
            
            if not df_jogos_time.empty:
                gols_feitos = df_jogos_time.apply(lambda x: x['home_score'] if x['home_team'] == time_escolhido else x['away_score'], axis=1).sum()
                gols_sofridos = df_jogos_time.apply(lambda x: x['away_score'] if x['home_team'] == time_escolhido else x['home_score'], axis=1).sum()
            else:
                gols_feitos = 0
                gols_sofridos = 0
            
            valores_score = df_consenso[df_consenso['team'] == time_escolhido]['predicted_score'].values
            score_2026 = valores_score[0] if len(valores_score) > 0 else 0
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Partidas Históricas", f"{total_jogos}")
            c2.metric("Taxa de Vitória", f"{taxa_vitoria:.1f}%")
            c3.metric("Gols Marcados", f"{int(gols_feitos)}")
            c4.metric("Gols Sofridos", f"{int(gols_sofridos)}")
            c5.metric("Força Prevista (2026)", f"{score_2026:.2f}/3.00")
            
            st.markdown("---")
            st.markdown("##### Histórico de Gols e Evolução (Últimos 20 anos)")
            
            df_historico = df_full[(df_full['team'] == time_escolhido) & (df_full['year'] >= 2005)].sort_values('year')
            if not df_historico.empty:
                fig_linha = go.Figure()
                
                # 1. Linha original azul contínua (Performance Real)
                fig_linha.add_trace(go.Scatter(
                    x=df_historico['year'], y=df_historico['performance_score'], 
                    mode='lines+markers', name='Performance Real', 
                    line=dict(color='#1f77b4', width=3)
                ))
                
                # 2. Pega as coordenadas do último ponto real para conectar a linha
                ultimo_ano = df_historico['year'].iloc[-1]
                ultimo_score = df_historico['performance_score'].iloc[-1]
                
                # 3. Linha tracejada cinza conectando o último ano real até 2026
                fig_linha.add_trace(go.Scatter(
                    x=[ultimo_ano, 2026], y=[ultimo_score, score_2026], 
                    mode='lines+markers', name='Previsão 2026 (IA)', 
                    line=dict(color='gray', width=2, dash='dash'), 
                    marker=dict(color='gray', size=8, symbol='circle') # Bolinha cinza
                ))
                
                fig_linha.update_layout(
                    xaxis_title="Ano", yaxis_title="Performance Escalonada (0 a 3)", 
                    height=350, yaxis=dict(range=[-0.2, 3.2]), 
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_linha, width='stretch')

        with tab_ano:
            st.markdown("### Desempenho Anual")
            col_a1, col_a2 = st.columns([1, 3])
            
            with col_a1:
                # Pega todos os anos do histórico
                anos_historicos = sorted(df_full['year'].unique(), reverse=True)
                ano_escolhido = st.selectbox("1. Selecione o Ano:", anos_historicos)
                
                # Filtra os times que jogaram naquele ano específico
                df_ano_filtrado = df_full[df_full['year'] == ano_escolhido]
                times_do_ano = sorted(df_ano_filtrado['team'].unique())
                
                idx_padrao_ano = times_do_ano.index("Brazil") if "Brazil" in times_do_ano else 0
                time_ano_escolhido = st.selectbox("2. Selecione a Seleção:", times_do_ano, index=idx_padrao_ano)
                
            with col_a2:
                st.markdown(f"#### Dados de {time_ano_escolhido} em {ano_escolhido}")
                
                # Busca os jogos brutos daquele time naquele exato ano
                df_bruto['year_temp'] = pd.to_datetime(df_bruto['date']).dt.year
                jogos_brutos_ano = df_bruto[((df_bruto['home_team'] == time_ano_escolhido) | (df_bruto['away_team'] == time_ano_escolhido)) & (df_bruto['year_temp'] == ano_escolhido)]
                
                if not jogos_brutos_ano.empty:
                    jogos = len(jogos_brutos_ano)
                    vitorias = len(jogos_brutos_ano[((jogos_brutos_ano['home_team'] == time_ano_escolhido) & (jogos_brutos_ano['home_score'] > jogos_brutos_ano['away_score'])) | ((jogos_brutos_ano['away_team'] == time_ano_escolhido) & (jogos_brutos_ano['away_score'] > jogos_brutos_ano['home_score']))])
                    empates = len(jogos_brutos_ano[jogos_brutos_ano['home_score'] == jogos_brutos_ano['away_score']])
                    derrotas = jogos - vitorias - empates
                    
                    gols_feitos_ano = jogos_brutos_ano.apply(lambda x: x['home_score'] if x['home_team'] == time_ano_escolhido else x['away_score'], axis=1).sum()
                    gols_sofridos_ano = jogos_brutos_ano.apply(lambda x: x['away_score'] if x['home_team'] == time_ano_escolhido else x['home_score'], axis=1).sum()
                    aproveitamento = ((vitorias * 3) + empates) / (jogos * 3) * 100

                    ca1, ca2, ca3, ca4, ca5 = st.columns(5)
                    ca1.metric("Partidas Disputadas", f"{jogos}")
                    ca2.metric("V - E - D", f"{vitorias} - {empates} - {derrotas}")
                    ca3.metric("Aproveitamento", f"{aproveitamento:.1f}%")
                    ca4.metric("Gols Pró", f"{int(gols_feitos_ano)}")
                    ca5.metric("Gols Contra", f"{int(gols_sofridos_ano)}")
                    
                    st.markdown("##### Lista Oficial de Partidas")
                    
                    # Prepara uma tabela bonita e amigável para o usuário ler
                    df_exibicao = jogos_brutos_ano[['date', 'tournament', 'home_team', 'home_score', 'away_score', 'away_team', 'neutral']].copy()
                    df_exibicao['date'] = pd.to_datetime(df_exibicao['date']).dt.strftime('%d/%m/%Y')
                    df_exibicao['Placar'] = df_exibicao['home_score'].astype(int).astype(str) + " x " + df_exibicao['away_score'].astype(int).astype(str)
                    df_exibicao['Local'] = np.where(df_exibicao['neutral'], "Campo Neutro", np.where(df_exibicao['home_team'] == time_ano_escolhido, "Em Casa", "Visitante"))
                    
                    df_exibicao = df_exibicao[['date', 'tournament', 'home_team', 'Placar', 'away_team', 'Local']]
                    df_exibicao.columns = ['Data', 'Torneio', 'Mandante', 'Placar', 'Visitante', 'Mando de Campo']
                    
                    st.dataframe(df_exibicao, hide_index=True)
                else:
                    st.warning("Esta seleção não disputou partidas oficiais registradas neste ano.")
    else:
        st.error("Não foram encontradas previsões para 2026 no banco de dados. Verifique a geração do banco.")

# MÓDULO 2: PAINEL DE DATA SCIENCE E TÉCNICAS
elif modo_app == "Visão Técnica":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configuração da Simulação")
    
    lista_modelos_disponiveis = ["ARIMA", "Exponencial", "LSTM", "Prophet", "Random Forest", "XGBoost"]
    modelos_escolhidos = st.sidebar.multiselect("Escolha as Técnicas:", options=lista_modelos_disponiveis, default=["Random Forest", "XGBoost"])

    anos_com_gabarito = df_full['year'].unique()
    anos_disponiveis = [int(ano) for ano in sorted(df_previsoes_banco['predicted_year'].unique()) if ano in anos_com_gabarito]
    
    if not anos_disponiveis:
        st.error("Nenhum dado encontrado no banco que possa ser comparado com a realidade.")
        st.stop()
        
    target_year = st.sidebar.selectbox("Escolha o Ano a ser analisado:", options=anos_disponiveis, index=len(anos_disponiveis)-1)
    horizonte_anos = st.sidebar.slider("Anos de Gap:", min_value=1, max_value=10, value=5)

    cutoff_year = target_year - horizonte_anos 
    lista_anos_alvo = list(range(target_year - 4, target_year + 1))

    st.sidebar.markdown(f"""
    ---
    **Resumo do Benchmark:**
    * **Treinamento:** Até {cutoff_year}
    * **Gap:** {horizonte_anos} anos
    * **Avaliação:** {target_year}
    """)

    st.title(f"Benchmarking de Modelos: Previsões de {target_year}")

    if not modelos_escolhidos:
        st.warning("Por favor, selecione pelo menos um modelo na barra lateral para começar.")
        st.stop()

    dict_resultados = {}
    for mod in modelos_escolhidos:
        filtro = (df_previsoes_banco['model'] == mod) & ((df_previsoes_banco['predicted_year'] - df_previsoes_banco['cutoff_year']) == horizonte_anos) & (df_previsoes_banco['predicted_year'].isin(lista_anos_alvo))
        df_mod_preds = df_previsoes_banco[filtro]
        if not df_mod_preds.empty:
            df_res = pd.merge(df_mod_preds, df_full, left_on=['team', 'predicted_year'], right_on=['team', 'year'], how='inner')
            dict_resultados[mod] = df_res

    if not dict_resultados:
        st.error(f"Nenhum dado pré-calculado encontrado para o Gap de {horizonte_anos} anos no corte de {cutoff_year}.")
        st.stop()

    st.subheader("Qual técnica previu melhor?")
    dados_tabela = []
    for mod, df_m in dict_resultados.items():
        df_m_target = df_m[df_m['year'] == target_year]
        if not df_m_target.empty:
            mae = mean_absolute_error(df_m_target['performance_score'], df_m_target['predicted_score'])
            rmse = np.sqrt(mean_squared_error(df_m_target['performance_score'], df_m_target['predicted_score']))
            corr = df_m_target['performance_score'].corr(df_m_target['predicted_score']) if df_m_target['predicted_score'].nunique() > 1 else 0.0
            dados_tabela.append({"Técnica": mod, "Erro Médio (MAE) ↓": mae, "RMSE ↓": rmse, "Correlação (R) ↑": corr})

    if dados_tabela:
        df_leaderboard = pd.DataFrame(dados_tabela).set_index("Técnica")
        st.dataframe(df_leaderboard.style.highlight_min(subset=["Erro Médio (MAE) ↓", "RMSE ↓"], color='#90ee90')
                                       .highlight_max(subset=["Correlação (R) ↑"], color='#90ee90'), width='stretch')

    st.markdown("---")
    primeiro_modelo = list(dict_resultados.keys())[0]
    df_referencia = dict_resultados[primeiro_modelo]
    df_ref_target = df_referencia[df_referencia['year'] == target_year]
    times_no_ano = list(sorted(df_ref_target['team'].unique()))

    if 'time_selecionado' not in st.session_state or st.session_state['time_selecionado'] not in times_no_ano:
        st.session_state['time_selecionado'] = times_no_ano[0]

    indice_atual = times_no_ano.index(st.session_state['time_selecionado'])

    col_sel_1, col_sel_2 = st.columns([1, 3])
    with col_sel_1:
        novo_time = st.selectbox("Escolha seleção a ser analisada:", times_no_ano, index=indice_atual)
        if novo_time != st.session_state['time_selecionado']:
            st.session_state['time_selecionado'] = novo_time
            st.rerun()

    time_selecionado = st.session_state['time_selecionado']
    real_score = df_ref_target[df_ref_target['team'] == time_selecionado].iloc[0]['performance_score']

    st.subheader(f"{time_selecionado} (Real: {real_score:.2f})")

    tab1, tab2, tab3 = st.tabs(["Comparativo Entre Técnicas", "Dispersão Global", "Evolução Temporal"])

    with tab1:
        fig_simples = go.Figure()
        fig_simples.add_trace(go.Bar(name="Gabarito", x=['<b>Pontuação<br>Real</b>'], y=[real_score], marker_color='#1f77b4', text=[f"{real_score:.2f}"], textposition='inside', textfont=dict(size=16, color='white'), hovertemplate="Real: %{y:.2f}<extra></extra>"))
        
        previsoes_ordenadas = []
        for mod in modelos_escolhidos:
            df_m_target = dict_resultados[mod][dict_resultados[mod]['year'] == target_year]
            if time_selecionado in df_m_target['team'].values:
                pred = df_m_target[df_m_target['team'] == time_selecionado].iloc[0]['predicted_score']
                distancia = abs(pred - real_score) 
                previsoes_ordenadas.append({'modelo': mod, 'predicao': pred, 'distancia': distancia})

        previsoes_ordenadas = sorted(previsoes_ordenadas, key=lambda k: k['distancia'])

        for item in previsoes_ordenadas:
            mod = item['modelo']
            pred = item['predicao']
            fig_simples.add_trace(go.Bar(name=mod, x=[f"{mod}<br>"], y=[pred], marker_color='lightgray', text=[f"{pred:.2f}"], textposition='inside', textfont=dict(size=14, color='black'), hovertemplate=f"{mod}: %{{y:.2f}}<extra></extra>"))

        fig_simples.update_layout(height=400, yaxis=dict(range=[0, 3.2], showticklabels=False, title=""), xaxis=dict(title="", tickangle=0), margin=dict(l=20, r=20, t=30, b=40), showlegend=False, barmode='group')
        st.plotly_chart(fig_simples, width='stretch')
        
        st.markdown("##### Dados Históricos Utilizados")
        df_team = df_ref_target[df_ref_target['team'] == time_selecionado].iloc[0]
        st.table(pd.DataFrame({'Métrica': ['Performance No Ano Anterior', 'Performance de 2 Anos Atrás', 'Média De Gols Marcados No Ano Anterior', 'Média De Gols Marcados 2 Anos Atrás'], 'Valor': [f"{df_team['prev_score_1']:.4f}", f"{df_team['prev_score_2']:.4f}", f"{df_team['prev_goals_1']:.2f}", f"{df_team['prev_goals_2']:.2f}"]}).set_index('Métrica'))

    with tab2:
        st.markdown("<h4 style='margin-bottom: -15px;'><b> Clique em qualquer bolinha para analisar a seleção!</b></h4>", unsafe_allow_html=True)
        fig_scatter = go.Figure()
        cores_linhas = px.colors.qualitative.Plotly
        for i, mod in enumerate(modelos_escolhidos):
            df_m_target = dict_resultados[mod][dict_resultados[mod]['year'] == target_year]
            df_others = df_m_target[df_m_target['team'] != time_selecionado]
            fig_scatter.add_trace(go.Scatter(x=df_others['performance_score'], y=df_others['predicted_score'], mode='markers', name=mod, marker=dict(color=cores_linhas[i % len(cores_linhas)], size=6, opacity=0.4), text=df_others['team'], customdata=df_others['team'], legendgroup=mod, hovertemplate="<b>%{text}</b><br>Real: %{x:.2f}<br>Previsto: %{y:.2f}<extra></extra>"))
            df_sel = df_m_target[df_m_target['team'] == time_selecionado]
            if not df_sel.empty:
                fig_scatter.add_trace(go.Scatter(x=df_sel['performance_score'], y=df_sel['predicted_score'], mode='markers', name=f"{mod} ({time_selecionado})", marker=dict(color=cores_linhas[i % len(cores_linhas)], size=18, line=dict(width=3, color='black')), text=df_sel['team'], customdata=df_sel['team'], legendgroup=mod, hovertemplate=f"<b>{time_selecionado}</b><br>Real: %{{x:.2f}}<br>Previsto ({mod}): %{{y:.2f}}<extra></extra>"))
        fig_scatter.add_shape(type="line", line=dict(dash='dash', color='gray'), x0=0, y0=0, x1=3, y1=3)
        fig_scatter.update_layout(xaxis_title="Performance Real", yaxis_title="Performance Prevista", height=500, clickmode='event+select', margin=dict(t=30))
        evento_clique = st.plotly_chart(fig_scatter, width='stretch', on_select="rerun", selection_mode="points")
        if evento_clique and len(evento_clique.selection['points']) > 0:
            ponto = evento_clique.selection['points'][0]
            time_clicado = ponto.get('text', None)
            if time_clicado is None and 'customdata' in ponto: time_clicado = ponto['customdata'][0] if isinstance(ponto['customdata'], list) else ponto['customdata']
            time_clicado = str(time_clicado).strip("[]'\" ")
            if time_clicado in times_no_ano and time_clicado != st.session_state['time_selecionado']:
                st.session_state['time_selecionado'] = time_clicado
                st.rerun()

    with tab3:
        st.markdown(f"##### Variação de Previsões ({target_year - 4} a {target_year})")
        fig_linha = go.Figure()
        df_real_team = df_referencia[df_referencia['team'] == time_selecionado].sort_values('year')
        if not df_real_team.empty:
            fig_linha.add_trace(go.Scatter(x=df_real_team['year'], y=df_real_team['performance_score'], mode='lines+markers', name='Gabarito (Real)', line=dict(color='#1f77b4', width=3), marker=dict(size=8), hovertemplate="Ano: %{x}<br>Real: %{y:.2f}<extra></extra>"))
        cores_linhas_evolucao = px.colors.qualitative.Pastel
        for i, mod in enumerate(modelos_escolhidos):
            df_m_team = dict_resultados[mod][dict_resultados[mod]['team'] == time_selecionado].sort_values('year')
            if not df_m_team.empty:
                fig_linha.add_trace(go.Scatter(x=df_m_team['year'], y=df_m_team['predicted_score'], mode='lines+markers', name=mod, line=dict(color=cores_linhas_evolucao[i % len(cores_linhas_evolucao)], width=2, dash='dot'), marker=dict(size=6), hovertemplate=f"Ano: %{{x}}<br>{mod}: %{{y:.2f}}<extra></extra>"))
        fig_linha.update_layout(xaxis_title="Ano", yaxis_title="Performance Escalonada", height=450, yaxis=dict(range=[-0.2, 3.2]), xaxis=dict(tickmode='linear', dtick=1), margin=dict(l=20, r=20, t=30, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_linha, width='stretch')
