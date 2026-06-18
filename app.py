import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

class DataAnalyzer:
    """Encapsula estadísticas descriptivas, clasificación de variables y visualizaciones."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clasificar_variables(self) -> dict:
        """Función personalizada que clasifica variables en numéricas y categóricas."""
        numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categoricas = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        return {"numericas": numericas, "categoricas": categoricas}

    def estadisticas_descriptivas(self) -> pd.DataFrame:
        return self.df.describe()

    def valores_faltantes(self) -> pd.Series:
        return self.df.isnull().sum()

    def conteo_categorica(self, col: str) -> pd.Series:
        return self.df[col].value_counts()

    def media(self, col: str) -> float:
        return self.df[col].mean()

    def mediana(self, col: str) -> float:
        return self.df[col].median()

    def moda(self, col: str):
        return self.df[col].mode()[0]

    def distribucion(self, col: str) -> pd.Series:
        return self.df[col].value_counts(normalize=True) * 100

    def comparacion_grupos(self, num_col: str, cat_col: str) -> pd.DataFrame:
        return self.df.groupby(cat_col)[num_col].describe()


st.set_page_config(
    page_title="Bank Marketing EDA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    [data-testid="stSidebar"] {background-color: #0d1b2a;}
    [data-testid="stSidebar"] * {color: #e0e6ef !important;}
    .main-title {font-size:2.2rem; font-weight:800; color:#1a73e8; margin-bottom:0;}
    .sub-title  {font-size:1rem;   color:#6c757d; margin-top:0;}
    .metric-card {
        background: #f0f4ff;
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .conclusion-box {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=70)
    st.markdown("## 🏦 Bank Marketing EDA")
    st.markdown("---")
    modulo = st.selectbox(
        "📂 Navegar a:",
        ["🏠 Home", "📂 Carga del Dataset", "🔍 Análisis EDA", "✅ Conclusiones"],
    )
    st.markdown("---")
    st.markdown("**Tecnologías utilizadas:**")
    st.markdown("🐍 Python · 🐼 Pandas · 📊 Seaborn")
    st.markdown("📈 Matplotlib · 🚀 Streamlit · 🔢 NumPy")



if "df" not in st.session_state:
    st.session_state.df = None




# ════════════════════════════════════════════════════════════
# MÓDULO 1 – HOME
# ════════════════════════════════════════════════════════════
if modulo == "🏠 Home":
    st.markdown('<p class="main-title">🏦 Análisis Exploratorio de Datos</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Dataset: Bank Marketing Campaign · Institución Financiera</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎯 Objetivo del Análisis")
        st.markdown("""
        Este proyecto realiza un **Análisis Exploratorio de Datos (EDA)** sobre el dataset
        **BankMarketing.csv**, correspondiente a una institución financiera que busca entender
        los factores que influyen en la aceptación de sus campañas de marketing.

        Durante los últimos 6 meses, la efectividad cayó del **12 % al 8 %**, afectando
        los bonos de los ejecutivos comerciales. La tarea consiste en analizar los datos
        de la última campaña para **descubrir relaciones y comportamientos relevantes**
        entre las variables.
        """)

        st.markdown("### 👤 Datos del Autor")
        st.markdown("""
        | Campo | Detalle |
        |---|---|
        | **Nombre completo** | Estudiante Analítica |
        | **Curso / Especialización** | Especialización en Python for Analytics |
        | **Año** | 2026 |
        """)

    with col2:
        st.markdown("### 📋 Sobre el Dataset")
        st.markdown("""
        <div class="metric-card">
        <b>Fuente:</b> Institución Financiera<br>
        <b>Registros:</b> ~4 500 clientes<br>
        <b>Variables:</b> 21 columnas<br>
        <b>Variable objetivo:</b> <code>y</code> (aceptó la campaña)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Tecnologías")
        techs = ["Python 3.x", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Streamlit"]
        for t in techs:
            st.markdown(f"✅ {t}")

    st.markdown("---")
    st.info("👈 Usa el menú lateral para navegar entre módulos. Comienza cargando el dataset.")





