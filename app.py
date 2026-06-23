import streamlit as st
import lightgbm as lgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. Configuração da Página
# ==========================================
st.set_page_config(
    page_title="Simulador Clínico | Parkinson", page_icon="🧬", layout="wide"
)


# ==========================================
# 2. Carregamento do Modelo e Constantes
# ==========================================
@st.cache_resource
def load_model():
    # Carrega o modelo matemático exportado pelo Jupyter Notebook (Versão V2 com 10 features)
    return lgb.Booster(model_file="lightgbm_clinico_app.txt")


model = load_model()
RMSE_MODELO = (
    9.11  # O erro absoluto médio validado na versão final com engenharia de features
)

# ==========================================
# 3. Cabeçalho Principal
# ==========================================
st.title("🧬 Simulador Clínico - Doença de Parkinson")
st.markdown("Motor Preditivo Longitudinal com Base no Protocolo MDS-UPDRS")
st.divider()

# ==========================================
# 4. Interface Lateral (Entrada de Dados e Lógica Condicional)
# ==========================================
with st.sidebar:
    st.header("Avaliação Basal (Mês 0)")
    st.markdown("Insira os escores exatos do diagnóstico inicial:")

    updrs_1 = st.number_input(
        "UPDRS Parte 1 (Cognição/Humor)", min_value=0, max_value=52, value=5, step=1
    )
    updrs_2 = st.number_input(
        "UPDRS Parte 2 (Atividades Diárias)", min_value=0, max_value=52, value=6, step=1
    )
    updrs_3 = st.number_input(
        "UPDRS Parte 3 (Exame Motor Inicial)",
        min_value=0,
        max_value=132,
        value=15,
        step=1,
    )

    st.divider()
    st.header("Parâmetros de Projeção")

    visit_month = st.slider(
        "Horizonte de Projeção (Meses):", min_value=12, max_value=120, value=60, step=12
    )
    med_status = st.radio(
        "Estratégia Farmacológica:", options=["Sem Medicação", "Com Medicação"]
    )

    # Lógica Dinâmica e Clínico-Sustentável:
    # O início do tratamento e o escore UPDRS 4 (complicações motoras induzidas por remédio)
    # só existem se o status da medicação for ativo. Caso contrário, são travados em zero.
    mes_inicio_med = 0
    updrs_4 = 0

    if med_status == "Com Medicação":
        mes_inicio_med = st.slider(
            "Mês de Início do Tratamento:",
            min_value=0,
            max_value=visit_month,
            value=0,
            step=12,
            help="O modelo calculará a progressão natural até este mês e, em seguida, aplicará o efeito da medicação.",
        )

        st.divider()
        st.markdown("### Histórico Terapêutico")
        updrs_4 = st.number_input(
            "UPDRS Parte 4 (Complicações Motoras)",
            min_value=0,
            max_value=24,
            value=0,
            step=1,
            help="Escore de complicações (discinesias) presentes na avaliação inicial.",
        )

# ==========================================
# 5. Motor Preditivo (Sincronização com Engenharia de Features V2)
# ==========================================
# Gerando o vetor de meses para traçar a linha do tempo contínua
meses_trajetoria = list(range(12, visit_month + 12, 12))

# Mapeamento do status da medicação ao longo da linha do tempo do paciente
medication_array = []
for mes in meses_trajetoria:
    if med_status == "Com Medicação" and mes >= mes_inicio_med:
        medication_array.append(1)
    else:
        medication_array.append(0)

# Construção da matriz base
dados_simulacao = pd.DataFrame(
    {
        "visit_month": meses_trajetoria,
        "medication_on": medication_array,
        "updrs_1_baseline": updrs_1,
        "updrs_2_baseline": updrs_2,
        "updrs_3_baseline": updrs_3,
        "updrs_4_baseline": updrs_4,
    }
)

# --- ENGENHARIA DE FEATURES EM TEMPO REAL ---
# Recriando exatamente as mesmas interações matemáticas que o modelo aprendeu no treino
dados_simulacao["updrs_total_baseline"] = updrs_1 + updrs_2 + updrs_3 + updrs_4
dados_simulacao["motor_adl_ratio"] = updrs_3 / (updrs_2 + 1.0)
dados_simulacao["motor_cog_ratio"] = updrs_3 / (updrs_1 + 1.0)
dados_simulacao["baseline_time_interaction"] = updrs_3 * dados_simulacao["visit_month"]

# Garantindo a ordem idêntica das colunas estruturadas no LightGBM
features_ordenadas = [
    "visit_month",
    "medication_on",
    "updrs_1_baseline",
    "updrs_2_baseline",
    "updrs_3_baseline",
    "updrs_4_baseline",
    "updrs_total_baseline",
    "motor_adl_ratio",
    "motor_cog_ratio",
    "baseline_time_interaction",
]
dados_simulacao = dados_simulacao[features_ordenadas]

# Execução do prognóstico
previsoes_trajetoria = np.asarray(model.predict(dados_simulacao)).flatten()
previsao_final = float(previsoes_trajetoria[-1])

# ==========================================
# 6. Interface Principal (Painel de Métricas)
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Diagnóstico Basal (Mês 0)", value=f"{updrs_3} pontos")
with col2:
    st.metric(
        label=f"Projeção Final (Mês {visit_month})",
        value=f"{previsao_final:.1f} pontos",
        delta=f"{previsao_final - updrs_3:.1f} pts (Variação)",
        delta_color="inverse",
    )
with col3:
    st.metric(
        label="Margem de Erro Esperada (RMSE)",
        value=f"± {RMSE_MODELO:.2f} pontos",
        help="Intervalo estatístico derivado da validação cruzada do modelo de aprendizado de máquina.",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 7. Gráfico Interativo com Plotly
# ==========================================
st.subheader("📈 Curva de Progressão Motora")

eixo_x = [0] + meses_trajetoria
eixo_y = [updrs_3] + previsoes_trajetoria.tolist()

# Definição das bandas do intervalo de confiança (bloqueando valores negativos e acima do teto da escala)
y_inferior = [max(0, y - RMSE_MODELO) for y in eixo_y]
y_superior = [min(132, y + RMSE_MODELO) for y in eixo_y]

fig = go.Figure()

# Camada 1: Faixa de Incerteza Estatística
fig.add_trace(
    go.Scatter(
        x=eixo_x + eixo_x[::-1],
        y=y_superior + y_inferior[::-1],
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Intervalo de Confiança (± 1 RMSE)",
    )
)

# Camada 2: Linha Principal de Projeção
fig.add_trace(
    go.Scatter(
        x=eixo_x,
        y=eixo_y,
        mode="lines+markers",
        line=dict(color="#2563EB", width=4),
        marker=dict(size=10, color="white", line=dict(width=2, color="#2563EB")),
        name="Previsão do modelo de aprendizado de máquina",
        hovertemplate="<b>Mês de Acompanhamento:</b> %{x}<br><b>Projeção MDS-UPDRS 3:</b> %{y:.1f} pontos<extra></extra>",
    )
)

# Camada 3: Indicador de Linha de Intervenção Terapêutica
if med_status == "Com Medicação" and mes_inicio_med > 0:
    fig.add_vline(
        x=mes_inicio_med, line_width=2, line_dash="dash", line_color="#10B981"
    )
    fig.add_annotation(
        x=mes_inicio_med,
        y=max(eixo_y) + 4,
        text="Início da Intervenção Farmacológica",
        showarrow=False,
        font=dict(color="#10B981", size=12, family="Inter"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#10B981",
        borderwidth=1,
        borderpad=4,
    )

# Estilização Avançada do Layout
fig.update_layout(
    xaxis_title="Meses desde o Diagnóstico Inicial",
    yaxis_title="Pontuação Motora (MDS-UPDRS Parte 3)",
    hovermode="x unified",
    template="plotly_white",
    height=480,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# Travas inteligentes de eixos para evitar distorções visuais
fig.update_yaxes(range=[max(0, min(y_inferior) - 5), min(132, max(y_superior) + 15)])
fig.update_xaxes(tickvals=eixo_x)

# Renderização do componente digital interactivo
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 8. Rodapé
# ==========================================
st.divider()
st.caption(
    "Aviso Médico: Ferramenta de suporte à decisão clínica baseada em modelagem preditiva. A faixa sombreada indica a incerteza estatística calculada pelo modelo de aprendizado de máquina. O prognóstico final depende exclusivamente da avaliação médica soberana."
)
