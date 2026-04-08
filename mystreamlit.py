import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------
st.set_page_config(
    page_title="Predicción de Depósitos",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------
# ESTILOS PERSONALIZADOS — TEMA OSCURO VERDE/MAGENTA
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Base ── */
    .stApp {
        background-color: #0d0f14;
        background-image:
            radial-gradient(ellipse at 15% 20%, rgba(0, 200, 120, 0.07) 0%, transparent 55%),
            radial-gradient(ellipse at 85% 75%, rgba(200, 0, 140, 0.07) 0%, transparent 55%);
    }

    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ── Hero ── */
    .hero-card {
        background: rgba(22, 27, 38, 0.92);
        border: 1px solid rgba(0, 200, 120, 0.22);
        border-radius: 24px;
        padding: 2rem 2rem 1.5rem 2rem;
        box-shadow: 0 0 40px rgba(0, 200, 120, 0.06), 0 12px 35px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
    }

    .mini-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        background: rgba(0, 200, 120, 0.12);
        border: 1px solid rgba(0, 200, 120, 0.35);
        color: #00c878;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.9rem;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f0f4ff;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #7a8ba8;
        margin-bottom: 0;
    }

    /* ── Section cards ── */
    .section-card {
        background: rgba(22, 27, 38, 0.9);
        border-radius: 22px;
        padding: 1.4rem 1.4rem 0.9rem 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.07);
        height: 100%;
    }

    .section-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #c800a8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }

    .section-desc {
        font-size: 0.92rem;
        color: #7a8ba8;
        margin-bottom: 1.1rem;
    }

    /* ── Form labels ── */
    div[data-testid="stNumberInput"] > label,
    div[data-testid="stSelectbox"] > label {
        font-weight: 600;
        color: #c2cfe0 !important;
        font-size: 0.88rem !important;
    }

    /* ── Inputs & selects ── */
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        background-color: rgba(10, 12, 20, 0.85) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #e8f0ff !important;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-testid="stNumberInput"] input:hover {
        border-color: rgba(0, 200, 120, 0.4) !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-testid="stNumberInput"] input:focus {
        border-color: #00c878 !important;
        box-shadow: 0 0 0 2px rgba(0, 200, 120, 0.18) !important;
    }

    /* ── Button ── */
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 16px;
        padding: 0.95rem 1.1rem;
        font-size: 1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c878 0%, #009a60 45%, #c800a8 100%);
        color: #fff;
        box-shadow: 0 6px 28px rgba(0, 200, 120, 0.22), 0 6px 28px rgba(200, 0, 168, 0.12);
        transition: 0.2s ease-in-out;
        letter-spacing: 0.02em;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 36px rgba(0, 200, 120, 0.28), 0 10px 36px rgba(200, 0, 168, 0.18);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Result card ── */
    .result-card {
        background: rgba(22, 27, 38, 0.95);
        border-radius: 22px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 1rem;
    }

    .result-card h3 {
        color: #f0f4ff !important;
    }

    /* ── Success / Error overrides ── */
    div[data-testid="stAlert"][data-baseweb="notification"] {
        border-radius: 14px !important;
    }

    /* success → verde */
    div[class*="stSuccess"] {
        background: rgba(0, 200, 120, 0.1) !important;
        border: 1px solid rgba(0, 200, 120, 0.35) !important;
        color: #00e890 !important;
        border-radius: 14px !important;
    }

    /* error → magenta */
    div[class*="stError"] {
        background: rgba(200, 0, 168, 0.1) !important;
        border: 1px solid rgba(200, 0, 168, 0.35) !important;
        color: #e040c8 !important;
        border-radius: 14px !important;
    }

    /* ── Expander ── */
    details {
        background: rgba(15, 18, 28, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 14px !important;
        color: #c2cfe0 !important;
    }

    /* ── Footer ── */
    .footer-note {
        text-align: center;
        color: #45556a;
        font-size: 0.86rem;
        margin-top: 1.8rem;
    }

    /* ── Selectbox dropdown options ── */
    [data-baseweb="popover"] {
        background: #161b26 !important;
        border: 1px solid rgba(0, 200, 120, 0.2) !important;
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# CABECERA
# -------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="mini-badge">Modelo SVM · Simulación comercial</div>
        <div class="hero-title">🏦 Simulador de campaña de depósitos</div>
        <p class="hero-subtitle">
            Introduce los datos del cliente y obtén una predicción clara sobre si
            contratará o no un depósito a plazo fijo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
def feature_engineering(df):
    df_proc = df.copy()
    df_proc['contactado_antes'] = np.where(df_proc['pdays'] == -1, 0, 1)
    df_proc['pdays'] = df_proc['pdays'].replace(-1, np.nan)
    df_proc['contact_unknown'] = (df_proc['contact'] == 'unknown').astype(int)
    df_proc['poutcome'] = df_proc['poutcome'].fillna('no_contact')
    df_proc['education'] = df_proc['education'].replace([None], np.nan)

    binary_map = {'no': 0, 'yes': 1}
    for col in ['housing', 'loan']:
        df_proc[col] = df_proc[col].map(binary_map)

    df_proc['balance'] = np.sign(df_proc['balance']) * np.log1p(np.abs(df_proc['balance']))

    if 'day' in df_proc.columns:
        df_proc = df_proc.drop(columns=['day'])

    return df_proc


# -------------------------------------------------
# CARGA DEL MODELO
# -------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_final.pkl')


try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


# -------------------------------------------------
# FORMULARIO
# -------------------------------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Perfil del cliente</div>
            <div class="section-desc">Datos demográficos y situación general.</div>
        """,
        unsafe_allow_html=True,
    )

    age = st.number_input("Edad", min_value=18, max_value=100, value=35)
    job = st.selectbox(
        "Trabajo",
        [
            'management', 'technician', 'entrepreneur', 'blue-collar', 'unknown',
            'retired', 'admin.', 'services', 'self-employed', 'unemployed',
            'housemaid', 'student'
        ],
    )
    marital = st.selectbox("Estado civil", ['married', 'single', 'divorced'])
    education = st.selectbox(
        "Educación",
        ['tertiary', 'secondary', 'unknown', 'primary', None],
        format_func=lambda x: 'Sin dato' if x is None else x,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Situación financiera</div>
            <div class="section-desc">Información económica y productos activos.</div>
        """,
        unsafe_allow_html=True,
    )

    balance = st.number_input("Saldo anual medio (€)", value=1500)
    housing = st.selectbox("¿Tiene hipoteca?", ['yes', 'no'])
    loan = st.selectbox("¿Tiene préstamo personal?", ['no', 'yes'])
    contact = st.selectbox("Medio de contacto", ['cellular', 'unknown', 'telephone'])

    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Historial de campaña</div>
            <div class="section-desc">Últimos contactos y resultados anteriores.</div>
        """,
        unsafe_allow_html=True,
    )

    month = st.selectbox(
        "Mes",
        ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    )
    duration = st.number_input("Duración última llamada (seg)", min_value=0, value=120)
    campaign = st.number_input("Nº contactos esta campaña", min_value=1, value=2)
    pdays = st.number_input("Días desde campaña anterior", min_value=-1, value=-1)
    previous = st.number_input("Nº contactos previos", min_value=0, value=0)
    poutcome = st.selectbox(
        "Resultado campaña anterior",
        ['unknown', 'failure', 'other', 'success', None],
        format_func=lambda x: 'Sin dato' if x is None else x,
    )

    st.markdown("</div>", unsafe_allow_html=True)


day = 15

st.write("")

# -------------------------------------------------
# BOTÓN DE PREDICCIÓN
# -------------------------------------------------
predict = st.button("Realizar predicción 🚀")

# -------------------------------------------------
# RESULTADO
# -------------------------------------------------
if predict:
    datos_cliente = pd.DataFrame([
        {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome,
        }
    ])

    datos_proc = feature_engineering(datos_cliente)
    prediccion = modelo.predict(datos_proc)[0]

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("### Resultado de la predicción")

    if prediccion == 1 or prediccion == 'yes':
        st.success("✅ PREDICCIÓN: SÍ CONTRATARÁ")
        st.write(
            "El modelo de Máquina de Vectores de Soporte (SVM) estima que este cliente "
            "**sí contratará** el depósito a plazo fijo."
        )
    else:
        st.error("❌ PREDICCIÓN: NO CONTRATARÁ")
        st.write(
            "El modelo de Máquina de Vectores de Soporte (SVM) estima que este cliente "
            "**no contratará** el depósito a plazo fijo."
        )

    with st.expander("Ver datos procesados usados por el modelo"):
        st.dataframe(datos_proc, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer-note">
        Consejo: puedes desplegar esta app con Streamlit Cloud, Render o Hugging Face Spaces.
    </div>
    """,
    unsafe_allow_html=True,
)