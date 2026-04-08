import streamlit as st  # Librería para crear la app web
import pandas as pd     # Manejo de datos en tablas
import numpy as np      # Cálculos numéricos
import joblib           # Cargar el modelo entrenado

# -------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------
st.set_page_config(
    page_title="Predicción de Depósitos",  # Título de la pestaña
    page_icon="🏦",                        # Icono de la app
    layout="wide",                        # Layout ancho
    initial_sidebar_state="collapsed",    # Sidebar oculto por defecto
)

# -------------------------------------------------
# ESTILOS — TEMA OSCURO MAGENTA
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Fondo ── */
    .stApp {
        background-color: #0c0a10;
        background-image:
            radial-gradient(ellipse at 20% 15%, rgba(180,0,140,0.10) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(140,0,180,0.08) 0%, transparent 50%);
    }
    .main > div { padding-top: 2rem; padding-bottom: 2rem; }

    /* ── Hero ── */
    .hero-card {
        background: rgba(20,14,28,0.97);
        border: 1px solid rgba(200,0,160,0.30);
        border-radius: 20px;
        padding: 26px 28px 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 60px rgba(180,0,140,0.07);
    }
    .mini-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        background: rgba(200,0,160,0.13);
        border: 1px solid rgba(200,0,160,0.40);
        color: #e040c8;
        font-size: 11px; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.2rem; font-weight: 800;
        color: #f5eeff; line-height: 1.15; margin-bottom: 6px;
    }
    .hero-sub { font-size: 1rem; color: #7a6a92; margin-bottom: 0; }

    /* ── Section header (HTML puro, encima de cada columna) ── */
    .col-card {
        background: rgba(20,14,28,0.95);
        border-radius: 16px;
        border: 1px solid rgba(200,0,160,0.14);
        padding: 16px 18px 12px;
        margin-bottom: 6px;
    }
    .sec-title {
        font-size: 10px; font-weight: 700; color: #c800a0;
        letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 3px;
    }
    .sec-desc { font-size: 12.5px; color: #6a5a82; margin-bottom: 0; }

    /* ── Fondo de las columnas de Streamlit ── */
    [data-testid="column"] {
        background: rgba(20,14,28,0.80);
        border-radius: 18px;
        border: 1px solid rgba(200,0,160,0.10);
        padding: 0 14px 14px !important;
    }

    /* ── Labels ── */
    div[data-testid="stNumberInput"] > label,
    div[data-testid="stSelectbox"] > label {
        font-weight: 600 !important;
        color: #c4b0d8 !important;
        font-size: 0.87rem !important;
    }

    /* ── Inputs y selects ── */
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        background-color: rgba(12,8,18,0.90) !important;
        border-color: rgba(200,0,160,0.18) !important;
        color: #f0e8ff !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-testid="stNumberInput"] input:hover {
        border-color: rgba(200,0,160,0.45) !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-testid="stNumberInput"] input:focus {
        border-color: #c800a0 !important;
        box-shadow: 0 0 0 2px rgba(200,0,160,0.20) !important;
    }

    /* ── Botón ── */
    .stButton > button {
        width: 100%; border: none; border-radius: 14px;
        padding: 0.95rem; font-size: 1rem; font-weight: 700; color: #fff;
        background: linear-gradient(90deg, #a0009a 0%, #c800a0 50%, #8a00d4 100%);
        box-shadow: 0 6px 28px rgba(200,0,160,0.28);
        transition: 0.2s ease-in-out; letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 36px rgba(200,0,160,0.40);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ── Result card ── */
    .result-card {
        background: rgba(20,14,28,0.97);
        border-radius: 18px;
        border: 1px solid rgba(200,0,160,0.20);
        padding: 20px 24px;
        margin-top: 1.2rem;
        box-shadow: 0 0 40px rgba(180,0,140,0.06);
    }
    .result-card h3 { color: #f5eeff !important; }

    /* ── Alerta SÍ → violeta claro ── */
    div[class*="stSuccess"] {
        background: rgba(138,0,212,0.12) !important;
        border: 1px solid rgba(138,0,212,0.40) !important;
        color: #c87aff !important;
        border-radius: 12px !important;
    }

    /* ── Alerta NO → magenta ── */
    div[class*="stError"] {
        background: rgba(200,0,160,0.10) !important;
        border: 1px solid rgba(200,0,160,0.38) !important;
        color: #e040c8 !important;
        border-radius: 12px !important;
    }

    /* ── Expander ── */
    details {
        background: rgba(15,10,22,0.85) !important;
        border: 1px solid rgba(200,0,160,0.12) !important;
        border-radius: 12px !important;
        color: #c4b0d8 !important;
    }

    /* ── Footer ── */
    .footer-note {
        text-align: center; color: #4a3860;
        font-size: 0.86rem; margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,  # Permite usar HTML y CSS
)

# -------------------------------------------------
# CABECERA
# -------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="mini-badge">Modelo SVM · Simulación comercial</div>
        <div class="hero-title">🏦 Simulador de campaña de depósitos</div>
        <p class="hero-sub">
            Introduce los datos del cliente y obtén una predicción clara sobre si
            contratará o no un depósito a plazo fijo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,  # Renderiza HTML personalizado
)

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
def feature_engineering(df):
    df_proc = df.copy()  # Copia del dataframe original
    df_proc['contactado_antes'] = np.where(df_proc['pdays'] == -1, 0, 1)  # Si ha sido contactado antes
    df_proc['pdays'] = df_proc['pdays'].replace(-1, np.nan)  # Reemplaza -1 por NaN
    df_proc['contact_unknown'] = (df_proc['contact'] == 'unknown').astype(int)  # Marca contactos desconocidos
    df_proc['poutcome'] = df_proc['poutcome'].fillna('no_contact')  # Rellena valores nulos
    df_proc['education'] = df_proc['education'].replace([None], np.nan)  # Limpia educación
    binary_map = {'no': 0, 'yes': 1}  # Mapa binario
    for col in ['housing', 'loan']:
        df_proc[col] = df_proc[col].map(binary_map)  # Convierte a 0/1
    df_proc['balance'] = np.sign(df_proc['balance']) * np.log1p(np.abs(df_proc['balance']))  # Transformación logarítmica
    if 'day' in df_proc.columns:
        df_proc = df_proc.drop(columns=['day'])  # Elimina columna no usada
    return df_proc  # Devuelve datos procesados

# -------------------------------------------------
# CARGA DEL MODELO
# -------------------------------------------------
@st.cache_resource  # Cachea el modelo para no cargarlo varias veces
def cargar_modelo():
    return joblib.load('modelo_final.pkl')  # Carga el modelo desde archivo

try:
    modelo = cargar_modelo()  # Intenta cargar el modelo
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")  # Muestra error
    st.stop()  # Detiene la app si falla

# -------------------------------------------------
# FORMULARIO
# -------------------------------------------------
col1, col2, col3 = st.columns(3, gap="large")  # Crea 3 columnas

with col1:
    st.markdown('<div class="col-card"><div class="sec-title">Perfil del cliente</div><div class="sec-desc">Datos demográficos y situación general.</div></div>', unsafe_allow_html=True)
    age       = st.number_input("Edad", min_value=18, max_value=100, value=35)  # Edad
    job       = st.selectbox("Trabajo", ['management','technician','entrepreneur','blue-collar','unknown','retired','admin.','services','self-employed','unemployed','housemaid','student'])  # Tipo de trabajo
    marital   = st.selectbox("Estado civil", ['married','single','divorced'])  # Estado civil
    education = st.selectbox("Educación", ['tertiary','secondary','unknown','primary',None], format_func=lambda x: 'Sin dato' if x is None else x)  # Nivel educativo

with col2:
    st.markdown('<div class="col-card"><div class="sec-title">Situación financiera</div><div class="sec-desc">Información económica y productos activos.</div></div>', unsafe_allow_html=True)
    balance  = st.number_input("Saldo anual medio (€)", value=1500)  # Balance
    housing  = st.selectbox("¿Tiene hipoteca?", ['yes','no'])  # Hipoteca
    loan     = st.selectbox("¿Tiene préstamo personal?", ['no','yes'])  # Préstamo
    contact  = st.selectbox("Medio de contacto", ['cellular','unknown','telephone'])  # Tipo de contacto

with col3:
    st.markdown('<div class="col-card"><div class="sec-title">Historial de campaña</div><div class="sec-desc">Últimos contactos y resultados anteriores.</div></div>', unsafe_allow_html=True)
    month    = st.selectbox("Mes", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])  # Mes
    duration = st.number_input("Duración última llamada (seg)", min_value=0, value=120)  # Duración llamada
    campaign = st.number_input("Nº contactos esta campaña", min_value=1, value=2)  # Nº contactos
    pdays    = st.number_input("Días desde campaña anterior", min_value=-1, value=-1)  # Días desde contacto
    previous = st.number_input("Nº contactos previos", min_value=0, value=0)  # Contactos previos
    poutcome = st.selectbox("Resultado campaña anterior", ['unknown','failure','other','success',None], format_func=lambda x: 'Sin dato' if x is None else x)  # Resultado previo

day = 15  # Día fijo (no se pide al usuario)

# -------------------------------------------------
# BOTÓN
# -------------------------------------------------
predict = st.button("Realizar predicción 🚀")  # Botón de predicción

# -------------------------------------------------
# RESULTADO
# -------------------------------------------------
if predict:  # Si se pulsa el botón
    datos_cliente = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'balance': balance, 'housing': housing, 'loan': loan, 'contact': contact,
        'day': day, 'month': month, 'duration': duration, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
    }])  # Crea dataframe con los datos introducidos

    datos_proc = feature_engineering(datos_cliente)  # Procesa los datos
    prediccion = modelo.predict(datos_proc)[0]  # Realiza la predicción

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("### Resultado de la predicción")

    if prediccion == 1 or prediccion == 'yes':  # Si predice positivo
        st.success("✅ PREDICCIÓN: SÍ CONTRATARÁ")
        st.write("El modelo SVM estima que este cliente **sí contratará** el depósito a plazo fijo.")
    else:  # Si predice negativo
        st.error("❌ PREDICCIÓN: NO CONTRATARÁ")
        st.write("El modelo SVM estima que este cliente **no contratará** el depósito a plazo fijo.")

    with st.expander("Ver datos procesados usados por el modelo"):
        st.dataframe(datos_proc, use_container_width=True)  # Muestra datos procesados

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    '<div class="footer-note">Consejo: puedes desplegar esta app con Streamlit Cloud, Render o Hugging Face Spaces.</div>',
    unsafe_allow_html=True,  # Muestra nota final
)