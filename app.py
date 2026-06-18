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



# ════════════════════════════════════════════════════════════
# MÓDULO 2 – CARGA DEL DATASET
# ════════════════════════════════════════════════════════════
elif modulo == "📂 Carga del Dataset":
    st.markdown("## 📂 Carga del Dataset")
    st.markdown("Sube el archivo **BankMarketing.csv** para comenzar el análisis.")

    archivo = st.file_uploader("📁 Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo, sep=";")
            st.session_state.df = df
            st.success(f"✅ Archivo cargado correctamente: **{archivo.name}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("📋 Filas", df.shape[0])
            col2.metric("📊 Columnas", df.shape[1])
            col3.metric("🔢 Variables numéricas", df.select_dtypes(include=np.number).shape[1])

            st.markdown("### 👁️ Vista previa del dataset (primeras 5 filas)")
            st.dataframe(df.head(), use_container_width=True)

            st.markdown("### 📐 Dimensiones del dataset")
            st.markdown(f"El dataset contiene **{df.shape[0]} registros** y **{df.shape[1]} variables**.")
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")
    else:
        st.warning("⚠️ Ningún archivo cargado. Por favor, sube el CSV para continuar.")

# ════════════════════════════════════════════════════════════
# MÓDULO 3 – EDA
# ════════════════════════════════════════════════════════════
elif modulo == "🔍 Análisis EDA":
    if st.session_state.df is None:
        st.warning("⚠️ Primero debes cargar el dataset en el módulo **📂 Carga del Dataset**.")
        st.stop()

    df = st.session_state.df
    analyzer = DataAnalyzer(df)
    clases = analyzer.clasificar_variables()

    st.markdown("## 🔍 Análisis Exploratorio de Datos (EDA)")
    st.caption("Explora cada ítem usando las pestañas a continuación.")

    tabs = st.tabs([
        "📌 Info General",
        "🏷️ Clasificación",
        "📊 Estadísticas",
        "❓ Valores Nulos",
        "📈 Dist. Numéricas",
        "📉 Categóricas",
        "🔗 Bivariado Num",
        "🔗 Bivariado Cat",
        "🎛️ Análisis Dinámico",
        "💡 Hallazgos",
    ])

    # ── Ítem 1: Información general ──────────────────────────
    with tabs[0]:
        st.markdown("### 📌 Ítem 1 — Información general del dataset")
        st.markdown("Resumen técnico del DataFrame: tipos de datos, memoria y valores nulos.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Tipos de datos por columna")
            tipos_df = pd.DataFrame({"Tipo": df.dtypes, "No nulos": df.notnull().sum()})
            st.dataframe(tipos_df, use_container_width=True)
        with col2:
            st.markdown("#### Conteo de valores nulos")
            nulos = analyzer.valores_faltantes()
            fig, ax = plt.subplots(figsize=(5, 4))
            nulos.plot(kind="bar", color="#1a73e8", ax=ax)
            ax.set_title("Valores nulos por columna")
            ax.set_ylabel("Cantidad")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        buf = StringIO()
        df.info(buf=buf)
        st.markdown("#### .info() del dataset")
        st.code(buf.getvalue())

    # ── Ítem 2: Clasificación de variables ───────────────────
    with tabs[1]:
        st.markdown("### 🏷️ Ítem 2 — Clasificación de variables")
        st.markdown("Se usa una **función personalizada** dentro de la clase `DataAnalyzer` para clasificar las variables.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔢 Variables Numéricas")
            num_df = pd.DataFrame({"Variable": clases["numericas"], "Tipo": "Numérica"})
            st.dataframe(num_df, use_container_width=True)
            st.metric("Total numéricas", len(clases["numericas"]))
        with col2:
            st.markdown("#### 🔤 Variables Categóricas")
            cat_df = pd.DataFrame({"Variable": clases["categoricas"], "Tipo": "Categórica"})
            st.dataframe(cat_df, use_container_width=True)
            st.metric("Total categóricas", len(clases["categoricas"]))

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(
            [len(clases["numericas"]), len(clases["categoricas"])],
            labels=["Numéricas", "Categóricas"],
            autopct="%1.0f%%",
            colors=["#1a73e8", "#f9a825"],
            startangle=90,
        )
        ax.set_title("Proporción de tipos de variables")
        st.pyplot(fig)
        plt.close()

    # ── Ítem 3: Estadísticas descriptivas ────────────────────
    with tabs[2]:
        st.markdown("### 📊 Ítem 3 — Estadísticas descriptivas")
        st.markdown("Se emplea `.describe()` para obtener medidas de tendencia central y dispersión.")

        st.dataframe(analyzer.estadisticas_descriptivas().T, use_container_width=True)

        st.markdown("#### Interpretación básica")
        col1, col2, col3 = st.columns(3)
        col1.markdown('<div class="metric-card">📌 <b>age</b><br>Media: {:.1f} | Mediana: {:.0f}</div>'.format(
            analyzer.media("age"), analyzer.mediana("age")), unsafe_allow_html=True)
        col2.markdown('<div class="metric-card">⏱️ <b>duration</b><br>Media: {:.0f}s | Mediana: {:.0f}s</div>'.format(
            analyzer.media("duration"), analyzer.mediana("duration")), unsafe_allow_html=True)
        col3.markdown('<div class="metric-card">📞 <b>campaign</b><br>Media: {:.1f} | Moda: {}</div>'.format(
            analyzer.media("campaign"), analyzer.moda("campaign")), unsafe_allow_html=True)

    # ── Ítem 4: Valores faltantes ─────────────────────────────
    with tabs[3]:
        st.markdown("### ❓ Ítem 4 — Análisis de valores faltantes")
        nulos = analyzer.valores_faltantes()
        total_nulos = nulos.sum()

        if total_nulos == 0:
            st.success("✅ El dataset no presenta valores nulos directos.")
            st.markdown("""
            <div class="insight-box">
            ⚠️ <b>Discusión:</b> Aunque no existen <code>NaN</code>, la variable <code>pdays</code> usa
            el valor <b>999</b> como indicador de "cliente no contactado previamente". Esto debe considerarse
            en análisis posteriores para evitar distorsiones estadísticas.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ Se encontraron {total_nulos} valores nulos.")
            fig, ax = plt.subplots(figsize=(8, 3))
            nulos[nulos > 0].plot(kind="bar", color="#e53935", ax=ax)
            ax.set_title("Columnas con valores nulos")
            st.pyplot(fig)
            plt.close()

        st.markdown("#### Conteo completo de nulos por variable")
        st.dataframe(nulos.reset_index().rename(columns={"index": "Variable", 0: "Nulos"}), use_container_width=True)

    # ── Ítem 5: Distribución de variables numéricas ───────────
    with tabs[4]:
        st.markdown("### 📈 Ítem 5 — Distribución de variables numéricas")
        st.markdown("Histogramas para visualizar la distribución de cada variable numérica.")

        num_cols = clases["numericas"]
        col_sel = st.selectbox("Selecciona variable numérica:", num_cols)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(df[col_sel], bins=30, kde=True, color="#1a73e8", ax=ax)
            ax.set_title(f"Distribución de {col_sel}")
            ax.set_xlabel(col_sel)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(y=df[col_sel], color="#90caf9", ax=ax)
            ax.set_title(f"Boxplot de {col_sel}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("#### Todos los histogramas")
        mostrar_todos = st.checkbox("Ver todos los histogramas")
        if mostrar_todos:
            n = len(num_cols)
            cols_grid = 3
            rows_grid = (n + cols_grid - 1) // cols_grid
            fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(14, rows_grid * 3))
            axes = axes.flatten()
            for i, col in enumerate(num_cols):
                sns.histplot(df[col], bins=25, kde=True, ax=axes[i], color="#1a73e8")
                axes[i].set_title(col, fontsize=9)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Ítem 6: Variables categóricas ────────────────────────
    with tabs[5]:
        st.markdown("### 📉 Ítem 6 — Análisis de variables categóricas")
        cat_cols = clases["categoricas"]
        cat_sel = st.selectbox("Selecciona variable categórica:", cat_cols)

        conteo = analyzer.conteo_categorica(cat_sel)
        prop = analyzer.distribucion(cat_sel)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Conteo — `{cat_sel}`**")
            st.dataframe(
                pd.DataFrame({"Categoría": conteo.index, "Conteo": conteo.values, "% ": prop.values.round(1)}),
                use_container_width=True,
            )
        with col2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.barplot(x=conteo.values, y=conteo.index, palette="Blues_r", ax=ax)
            ax.set_title(f"Distribución de {cat_sel}")
            ax.set_xlabel("Frecuencia")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Ítem 7: Bivariado numérico vs categórico ─────────────
    with tabs[6]:
        st.markdown("### 🔗 Ítem 7 — Análisis bivariado (numérico vs categórico)")
        st.markdown("Comparación de una variable numérica respecto a la variable objetivo **`y`** u otras categóricas.")

        num_sel7 = st.selectbox("Variable numérica:", clases["numericas"], key="biv_num")
        cat_sel7 = st.selectbox("Variable categórica:", clases["categoricas"], key="biv_cat")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df, x=cat_sel7, y=num_sel7, palette="Set2", ax=ax)
            ax.set_title(f"{num_sel7} por {cat_sel7}")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.violinplot(data=df, x=cat_sel7, y=num_sel7, palette="Set3", ax=ax)
            ax.set_title(f"Violinplot: {num_sel7} vs {cat_sel7}")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("#### Estadísticas por grupo")
        st.dataframe(analyzer.comparacion_grupos(num_sel7, cat_sel7), use_container_width=True)

    # ── Ítem 8: Bivariado categórico vs categórico ───────────
    with tabs[7]:
        st.markdown("### 🔗 Ítem 8 — Análisis bivariado (categórico vs categórico)")

        cat_sel8a = st.selectbox("Variable 1:", clases["categoricas"], key="biv_cat8a")
        cat_sel8b = st.selectbox("Variable 2 (referencia):", clases["categoricas"],
                                 index=clases["categoricas"].index("y") if "y" in clases["categoricas"] else 0,
                                 key="biv_cat8b")

        tabla = pd.crosstab(df[cat_sel8a], df[cat_sel8b], normalize="index") * 100

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            tabla.plot(kind="bar", stacked=True, colormap="tab10", ax=ax)
            ax.set_title(f"{cat_sel8a} vs {cat_sel8b} (%)")
            ax.set_ylabel("Porcentaje")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(tabla.round(1), annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title(f"Heatmap: {cat_sel8a} vs {cat_sel8b}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Ítem 9: Análisis dinámico ─────────────────────────────
    with tabs[8]:
        st.markdown("### 🎛️ Ítem 9 — Análisis basado en parámetros seleccionados")
        st.markdown("Selecciona variables para un análisis dinámico personalizado.")

        tipo_analisis = st.selectbox("Tipo de análisis:", ["Correlación entre numéricas", "Distribución filtrada por categoría"])

        if tipo_analisis == "Correlación entre numéricas":
            vars_sel = st.multiselect("Selecciona variables numéricas (mín. 2):", clases["numericas"],
                                      default=clases["numericas"][:5])
            if len(vars_sel) >= 2:
                fig, ax = plt.subplots(figsize=(8, 5))
                corr = df[vars_sel].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
                ax.set_title("Mapa de correlación")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Selecciona al menos 2 variables para la correlación.")

        else:
            col_num = st.selectbox("Variable numérica:", clases["numericas"], key="din_num")
            col_cat = st.selectbox("Variable categórica (filtro):", clases["categoricas"], key="din_cat")
            cats_disp = df[col_cat].unique().tolist()
            cats_sel = st.multiselect("Categorías a incluir:", cats_disp, default=cats_disp[:3])

            rango = st.slider(
                f"Filtrar rango de {col_num}:",
                float(df[col_num].min()),
                float(df[col_num].max()),
                (float(df[col_num].quantile(0.05)), float(df[col_num].quantile(0.95))),
            )
            mostrar_media = st.checkbox("Mostrar línea de media", value=True)

            df_fil = df[(df[col_cat].isin(cats_sel)) & (df[col_num].between(*rango))]

            fig, ax = plt.subplots(figsize=(8, 4))
            for cat in cats_sel:
                sub = df_fil[df_fil[col_cat] == cat][col_num]
                sns.kdeplot(sub, ax=ax, label=str(cat), fill=True, alpha=0.3)
            if mostrar_media:
                ax.axvline(df_fil[col_num].mean(), color="red", linestyle="--", label=f"Media: {df_fil[col_num].mean():.1f}")
            ax.set_title(f"Distribución de {col_num} por {col_cat}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(f"Registros filtrados: {len(df_fil):,}")

    # ── Ítem 10: Hallazgos clave ──────────────────────────────
    with tabs[9]:
        st.markdown("### 💡 Ítem 10 — Hallazgos clave del EDA")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Resumen visual — Variable objetivo `y`")
            conteo_y = df["y"].value_counts()
            fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
            axes[0].pie(conteo_y.values, labels=conteo_y.index, autopct="%1.1f%%",
                        colors=["#e53935", "#43a047"], startangle=90)
            axes[0].set_title("Aceptación de campaña")
            top_jobs = df[df["y"] == "yes"]["job"].value_counts().head(6)
            sns.barplot(x=top_jobs.values, y=top_jobs.index, palette="Greens_r", ax=axes[1])
            axes[1].set_title("Top empleos que aceptaron")
            axes[1].set_xlabel("Conteo")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### 🔍 Insights principales")
            insights = [
                "📌 Solo ~11% de los clientes aceptó la campaña (desbalanceo de clases).",
                "📌 La duración del contacto es el predictor más correlacionado con la aceptación.",
                "📌 Los clientes retirados (retired) y estudiantes tienen mayor tasa de conversión.",
                "📌 El mes de mayo concentra la mayor cantidad de contactos realizados.",
                "📌 Clientes sin crédito en mora tienen mayor probabilidad de aceptar.",
                "📌 El canal celular supera al teléfono fijo en efectividad.",
            ]
            for i in insights:
                st.markdown(f'<div class="insight-box">{i}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MÓDULO 4 – CONCLUSIONES
# ════════════════════════════════════════════════════════════
elif modulo == "✅ Conclusiones":
    st.markdown("## ✅ Conclusiones finales")
    st.markdown("Basadas en el análisis exploratorio del dataset BankMarketing.csv")
    st.markdown("---")

    conclusiones = [
        {
            "num": "1",
            "titulo": "La campaña tiene una tasa de conversión baja (~11%)",
            "texto": (
                "El dataset está altamente desbalanceado: aproximadamente el 89% de los clientes respondió 'no' "
                "a la campaña. Esto indica que la segmentación actual no es eficiente y se requiere un enfoque "
                "más dirigido para mejorar la tasa de conversión y recuperar el indicador de efectividad."
            ),
        },
        {
            "num": "2",
            "titulo": "La duración del contacto es el factor más influyente",
            "texto": (
                "Los clientes que eventualmente aceptaron la campaña tuvieron conversaciones significativamente "
                "más largas (mediana > 500s vs ~100s en rechazos). Aunque la duración no puede controlarse de "
                "antemano, es un indicador de interés real del cliente."
            ),
        },
        {
            "num": "3",
            "titulo": "El perfil demográfico importa: retirados y estudiantes lideran la conversión",
            "texto": (
                "Al analizar `job` vs `y`, los segmentos de jubilados y estudiantes muestran las tasas de "
                "aceptación más altas proporcionalmente, a pesar de no ser los grupos más contactados. "
                "Redirigir esfuerzos hacia estos segmentos podría mejorar la efectividad."
            ),
        },
        {
            "num": "4",
            "titulo": "El canal de comunicación impacta los resultados",
            "texto": (
                "Los contactos realizados por celular ('cellular') tienen mayor tasa de conversión que los "
                "realizados por teléfono fijo ('telephone'). La institución debería priorizar el canal móvil "
                "en futuras campañas para maximizar la efectividad."
            ),
        },
        {
            "num": "5",
            "titulo": "Los indicadores macroeconómicos correlacionan con la aceptación",
            "texto": (
                "Variables como `euribor3m`, `emp.var.rate` y `cons.conf.idx` muestran correlaciones "
                "negativas entre sí. Los periodos de menor tasa euribor coinciden con mayor aceptación, "
                "sugiriendo que el contexto económico favorable facilita la toma de decisiones financieras."
            ),
        },
    ]

    for c in conclusiones:
        st.markdown(f"""
        <div class="conclusion-box">
        <b>Conclusión {c['num']}: {c['titulo']}</b><br><br>
        {c['texto']}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### 🚀 Recomendaciones estratégicas")
    cols = st.columns(3)
    with cols[0]:
        st.info("🎯 **Segmentación** — Enfocar campañas en jubilados y estudiantes")
    with cols[1]:
        st.info("📱 **Canal** — Priorizar contacto celular sobre teléfono fijo")
    with cols[2]:
        st.info("📅 **Timing** — Aprovechar períodos de euribor bajo para campañas")



