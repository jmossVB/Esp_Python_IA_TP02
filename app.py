import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ═══════════════════════════════════════════════════════════════
# POO – clase DataAnalyzer
# Encapsula: estadísticas descriptivas, clasificación de variables
# y funciones de visualización (requisito POO del profe)
# ═══════════════════════════════════════════════════════════════
class DataAnalyzer:
    """
    Clase que encapsula el análisis exploratorio de datos.
    Atributos: DataFrame, listas de variables numéricas y categóricas.
    Métodos: clasificación, estadísticas, visualizaciones.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._clases = self._clasificar_variables()

    # ── Función personalizada de clasificación ──────────────────
    def _clasificar_variables(self) -> dict:
        """Clasifica automáticamente las variables del dataset."""
        numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categoricas = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        return {"numericas": numericas, "categoricas": categoricas}

    @property
    def numericas(self):
        return self._clases["numericas"]

    @property
    def categoricas(self):
        return self._clases["categoricas"]

    # ── Estadísticas ────────────────────────────────────────────
    def describe(self) -> pd.DataFrame:
        return self.df[self.numericas].describe()

    def media(self, col: str) -> float:
        return round(float(self.df[col].mean()), 2)

    def mediana(self, col: str) -> float:
        return round(float(self.df[col].median()), 2)

    def moda(self, col: str):
        return self.df[col].mode()[0]

    def dispersion(self, col: str) -> float:
        return round(float(self.df[col].std()), 2)

    def valores_nulos(self) -> pd.Series:
        return self.df.isnull().sum()

    def conteo_cat(self, col: str) -> pd.Series:
        return self.df[col].value_counts()

    def proporcion_cat(self, col: str) -> pd.Series:
        return (self.df[col].value_counts(normalize=True) * 100).round(2)

    def comparar_grupos(self, num_col: str, cat_col: str) -> pd.DataFrame:
        return self.df.groupby(cat_col)[num_col].agg(["mean", "median", "std", "count"]).round(2)

    # ── Visualizaciones (encapsuladas en la clase) ───────────────
    def plot_histograma(self, col: str, ax):
        sns.histplot(self.df[col], bins=30, kde=True, color="#1a73e8", ax=ax)
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)

    def plot_boxplot(self, col: str, ax):
        sns.boxplot(y=self.df[col], color="#90caf9", ax=ax)
        ax.set_title(f"Boxplot — {col}")

    def plot_barras_cat(self, col: str, ax):
        conteo = self.conteo_cat(col)
        sns.barplot(x=conteo.values, y=conteo.index, palette="Blues_r", ax=ax)
        ax.set_title(f"Conteos — {col}")
        ax.set_xlabel("Frecuencia")

    def plot_bivariado_num_cat(self, num_col: str, cat_col: str, ax):
        sns.boxplot(data=self.df, x=cat_col, y=num_col, palette="Set2", ax=ax)
        ax.set_title(f"{num_col} por {cat_col}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    def plot_heatmap_corr(self, cols: list, ax):
        corr = self.df[cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matriz de correlación")


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bank Marketing EDA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0d1b2a; }
    [data-testid="stSidebar"] * { color: #e0e6ef !important; }
    .card-azul {
        background: #f0f4ff;
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .card-amarillo {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .card-verde {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .card-rojo {
        background: #fdecea;
        border-left: 4px solid #c62828;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR – navegación
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Bank Marketing EDA")
    st.markdown("---")
    modulo = st.selectbox(
        "📂 Módulo:",
        [
            "🏠 Home",
            "📂 Carga del Dataset",
            "🔍 Análisis EDA",
            "✅ Conclusiones",
        ],
    )
    st.markdown("---")
    st.markdown("**Tecnologías:**")
    for t in ["🐍 Python 3.x", "🐼 Pandas", "🔢 NumPy", "📊 Seaborn", "📈 Matplotlib", "🚀 Streamlit"]:
        st.markdown(t)

# Session state
if "df" not in st.session_state:
    st.session_state.df = None


# ═══════════════════════════════════════════════════════════════
# MÓDULO 1 – HOME  (solo contextualiza, NO realiza análisis)
# ═══════════════════════════════════════════════════════════════
if modulo == "🏠 Home":
    st.title("🏦 Análisis Exploratorio de Datos — Bank Marketing")
    st.subheader("Especialización en Python for Analytics · Caso de Estudio N°1 — BankMarketing")
    st.markdown("---")
 
    col1, col2, col3 = st.columns([1, 2, 2])
 
    with col1:
        st.image("jmosquera.jpeg", width=200)
 
    with col2:
        st.markdown("### 👤 Información Personal")
        st.write("**Nombre:** José Alex Mosquera Amaro")
        st.write("**Linkedin:** https://www.linkedin.com/in/josemosquera/")
        st.write("**Año:** 2026")

        st.markdown("---")
        st.markdown("**Instructor:** MSc. Carlos Carrillo Villavicencio")
 
    with col3:
        st.markdown("Especialización en Python for Analytics")
        st.info("Caso de Estudio N°1 — Bank Marketing EDA")
        st.markdown("""
        <div class="card-azul">
        <b>Fuente:</b> Institución Financiera<br>
        <b>Registros:</b> ~4 500 clientes<br>
        <b>Variables:</b> 21 columnas<br>
        <b>Variable objetivo:</b> <code>y</code> (yes / no)
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("### 📋 Descripción del proyecto")
    st.markdown("""
    Este proyecto desarrolla una **aplicación interactiva construida en Python utilizando Streamlit**,
    orientada al **Análisis Exploratorio de Datos (EDA)** del dataset **BankMarketing.csv**,
    correspondiente a una institución financiera que busca entender los factores que influyen
    en la aceptación de sus campañas de marketing.
 
    Durante los últimos 6 meses, la efectividad *(e = Ventas/Base × 100%)* cayó del **12 % al 8 %**,
    afectando los bonos de los ejecutivos comerciales. La tarea consiste en analizar los datos de la
    última campaña para **descubrir relaciones y comportamientos relevantes** entre las variables.
 
    > ⚠️ El objetivo **NO es construir modelos predictivos**, sino aplicar de manera integrada
    los conceptos vistos a lo largo del curso, desarrollando una herramienta funcional, clara
    y bien estructurada, similar a un producto analítico real.
 
    La aplicación integra los siguientes conceptos fundamentales:
    - **Variables y tipos de datos · Funciones · f-strings**
    - **Programación Orientada a Objetos (POO)** — clase `DataAnalyzer`
    - **NumPy y Pandas** — manipulación y análisis de datos
    - **Visualización con Matplotlib y Seaborn**
    - **Estadística descriptiva** — media, mediana, moda, dispersión
    """)
 
    st.markdown("---")
 
    st.markdown("### 🛠️ Tecnologías utilizadas")
    tech = {
        "Tecnología": ["Python 3.x", "Streamlit", "Pandas", "NumPy", "Matplotlib", "Seaborn"],
        "Uso": [
            "Lenguaje base",
            "Interfaz interactiva (sidebar, tabs, columns, widgets)",
            "Manipulación y análisis de datos",
            "Cálculos numéricos y arrays",
            "Visualizaciones estadísticas base",
            "Gráficas estadísticas avanzadas",
        ],
    }
    st.dataframe(pd.DataFrame(tech), use_container_width=True, hide_index=True)
 
    st.markdown("---")
    st.info("👈 Navega usando el menú lateral. **Empieza cargando el dataset en 📂 Carga del Dataset.**")
 
 

# ═══════════════════════════════════════════════════════════════
# MÓDULO 2 – CARGA DEL DATASET
# ═══════════════════════════════════════════════════════════════
elif modulo == "📂 Carga del Dataset":
    st.title("📂 Carga del Dataset")
    st.markdown("Antes de ejecutar cualquier análisis, sube el archivo **BankMarketing.csv**.")
    st.markdown("---")

    archivo = st.file_uploader("📁 Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo, sep=";")
            st.session_state.df = df
            st.success(f"✅ Archivo **{archivo.name}** cargado correctamente.")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📋 Filas", f"{df.shape[0]:,}")
            col2.metric("📊 Columnas", df.shape[1])
            col3.metric("🔢 Vars. numéricas", df.select_dtypes(include=np.number).shape[1])
            col4.metric("🔤 Vars. categóricas", df.select_dtypes(include="object").shape[1])

            st.markdown("### 👁️ Vista previa — primeras 5 filas (head)")
            st.dataframe(df.head(), use_container_width=True)

            st.markdown("### 📐 Dimensiones del dataset")
            st.markdown(
                f"El dataset contiene **{df.shape[0]} registros** (filas) "
                f"y **{df.shape[1]} variables** (columnas)."
            )
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")
    else:
        st.warning("⚠️ Ningún archivo cargado. Sube el CSV para continuar.")
        st.markdown("""
        <div class="card-rojo">
        🚫 <b>Ningún análisis se ejecutará</b> hasta que el archivo sea cargado correctamente.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MÓDULO 3 – ANÁLISIS EDA (núcleo del proyecto)
# ═══════════════════════════════════════════════════════════════
elif modulo == "🔍 Análisis EDA":

    # Bloqueo si no hay dataset
    if st.session_state.df is None:
        st.warning("⚠️ Primero debes cargar el dataset en **📂 Carga del Dataset**.")
        st.markdown("""
        <div class="card-rojo">
        🚫 <b>Ningún análisis puede ejecutarse sin el archivo cargado.</b>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df
    da = DataAnalyzer(df)   # instancia de la clase POO

    st.title("🔍 Análisis Exploratorio de Datos (EDA)")
    st.caption("Este módulo es el núcleo del proyecto. Organizado en tabs y columns.")
    st.markdown("---")

    tabs = st.tabs([
        "📌 1. Info General",
        "🏷️ 2. Clasificación",
        "📊 3. Estadísticas",
        "❓ 4. Valores Nulos",
        "📈 5. Dist. Numéricas",
        "📉 6. Categóricas",
        "🔗 7. Bivariado Num-Cat",
        "🔗 8. Bivariado Cat-Cat",
        "🎛️ 9. Análisis Dinámico",
        "💡 10. Hallazgos",
    ])

    # ── Ítem 1 ────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 📌 Ítem 1 — Información general del dataset")
        st.markdown("Revisión técnica completa: `.info()`, tipos de datos y conteo de valores nulos.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Tipos de datos por columna")
            tipos_df = pd.DataFrame({
                "Columna": df.columns,
                "Tipo": df.dtypes.values,
                "No nulos": df.notnull().sum().values,
                "% completo": (df.notnull().mean() * 100).round(1).values,
            })
            st.dataframe(tipos_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Salida de `.info()`")
            buf = StringIO()
            df.info(buf=buf)
            st.code(buf.getvalue(), language=None)

        st.markdown("#### Conteo de valores nulos por columna")
        nulos = da.valores_nulos()
        fig, ax = plt.subplots(figsize=(10, 3))
        nulos.plot(kind="bar", color=["#c62828" if v > 0 else "#1a73e8" for v in nulos.values], ax=ax)
        ax.set_title("Valores nulos por columna")
        ax.set_ylabel("Cantidad de nulos")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Ítem 2 ────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### 🏷️ Ítem 2 — Clasificación de variables")
        st.markdown("""
        Se utiliza la **función personalizada** `_clasificar_variables()` dentro de la
        clase `DataAnalyzer` para identificar automáticamente variables numéricas y categóricas.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔢 Variables Numéricas")
            df_num = pd.DataFrame({
                "Variable": da.numericas,
                "Tipo Python": [str(df[c].dtype) for c in da.numericas],
                "Conteo no nulos": [df[c].notnull().sum() for c in da.numericas],
            })
            st.dataframe(df_num, use_container_width=True, hide_index=True)
            st.metric("Total numéricas", len(da.numericas))

        with col2:
            st.markdown("#### 🔤 Variables Categóricas")
            df_cat = pd.DataFrame({
                "Variable": da.categoricas,
                "Tipo Python": [str(df[c].dtype) for c in da.categoricas],
                "Categorías únicas": [df[c].nunique() for c in da.categoricas],
                "Conteo no nulos": [df[c].notnull().sum() for c in da.categoricas],
            })
            st.dataframe(df_cat, use_container_width=True, hide_index=True)
            st.metric("Total categóricas", len(da.categoricas))

        st.markdown("#### Proporción de tipos de variables")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(
            [len(da.numericas), len(da.categoricas)],
            labels=["Numéricas", "Categóricas"],
            autopct="%1.0f%%",
            colors=["#1a73e8", "#f9a825"],
            startangle=90,
        )
        ax.set_title("Distribución de tipos de variables")
        st.pyplot(fig)
        plt.close()

    # ── Ítem 3 ────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 📊 Ítem 3 — Estadísticas descriptivas")
        st.markdown("Uso de `.describe()` para obtener medidas de tendencia central y dispersión.")

        st.markdown("#### Tabla completa — `.describe()`")
        st.dataframe(da.describe().T.style.format("{:.2f}"), use_container_width=True)

        st.markdown("#### Interpretación: medias, medianas y dispersión")
        st.markdown("Selecciona una variable numérica para ver su análisis individual:")

        col_stat = st.selectbox("Variable numérica:", da.numericas, key="stat_col")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📐 Media", da.media(col_stat))
        col2.metric("📏 Mediana", da.mediana(col_stat))
        col3.metric("🔢 Moda", da.moda(col_stat))
        col4.metric("📉 Desv. estándar", da.dispersion(col_stat))

        diferencia = abs(da.media(col_stat) - da.mediana(col_stat))
        st.markdown(f"""
        <div class="card-azul">
        📌 <b>Interpretación de {col_stat}:</b><br>
        La diferencia entre media ({da.media(col_stat)}) y mediana ({da.mediana(col_stat)}) es <b>{diferencia:.2f}</b>.
        {'Una diferencia alta sugiere presencia de valores atípicos (outliers) que sesgan la distribución.' if diferencia > da.mediana(col_stat) * 0.2 else 'La distribución es relativamente simétrica, sin sesgo pronunciado.'}
        La desviación estándar de {da.dispersion(col_stat)} indica la dispersión promedio respecto a la media.
        </div>
        """, unsafe_allow_html=True)

    # ── Ítem 4 ────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### ❓ Ítem 4 — Análisis de valores faltantes")

        nulos = da.valores_nulos()
        total_nulos = nulos.sum()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de valores nulos", int(total_nulos))
            st.dataframe(
                nulos.reset_index().rename(columns={"index": "Variable", 0: "Nulos"}),
                use_container_width=True,
                hide_index=True,
            )
        with col2:
            if total_nulos == 0:
                st.success("✅ El dataset NO presenta valores nulos directos (NaN).")
            else:
                fig, ax = plt.subplots(figsize=(5, 4))
                nulos[nulos > 0].plot(kind="bar", color="#c62828", ax=ax)
                ax.set_title("Columnas con valores nulos")
                st.pyplot(fig)
                plt.close()

        st.markdown("""
        <div class="card-amarillo">
        ⚠️ <b>Discusión:</b> Aunque no existen valores <code>NaN</code>, la variable
        <code>pdays</code> usa el valor <b>999</b> como código especial que indica que el cliente
        <i>no fue contactado previamente</i>. Este valor puede distorsionar estadísticas descriptivas
        si no se trata adecuadamente. Se recomienda tratarlo como categoría separada en análisis posteriores.
        </div>
        """, unsafe_allow_html=True)

    # ── Ítem 5 ────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### 📈 Ítem 5 — Distribución de variables numéricas")
        st.markdown("Histogramas con KDE y boxplots usando Matplotlib y Seaborn.")

        col_num5 = st.selectbox("Variable numérica:", da.numericas, key="num5")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            da.plot_histograma(col_num5, ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            da.plot_boxplot(col_num5, ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(f"""
        <div class="card-azul">
        📌 <b>Interpretación visual — {col_num5}:</b><br>
        El histograma muestra la forma de la distribución. Si tiene una cola larga hacia la derecha,
        la distribución es sesgada positivamente. El boxplot permite identificar outliers (puntos fuera
        de los bigotes). La línea KDE superpuesta suaviza la distribución para facilitar su lectura.
        </div>
        """, unsafe_allow_html=True)

        ver_todos = st.checkbox("📊 Ver histogramas de todas las variables numéricas")
        if ver_todos:
            n = len(da.numericas)
            ncols = 3
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
            axes = axes.flatten()
            for i, col in enumerate(da.numericas):
                da.plot_histograma(col, axes[i])
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Ítem 6 ────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("### 📉 Ítem 6 — Análisis de variables categóricas")
        st.markdown("Conteos, gráficos de barras y proporciones por categoría.")

        col_cat6 = st.selectbox("Variable categórica:", da.categoricas, key="cat6")

        conteo = da.conteo_cat(col_cat6)
        prop = da.proporcion_cat(col_cat6)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### Conteo y proporción — `{col_cat6}`")
            df_resumen = pd.DataFrame({
                "Categoría": conteo.index,
                "Conteo": conteo.values,
                "Proporción (%)": prop.values,
            })
            st.dataframe(df_resumen, use_container_width=True, hide_index=True)

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            da.plot_barras_cat(col_cat6, ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Ítem 7 ────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("### 🔗 Ítem 7 — Análisis bivariado (numérico vs categórico)")
        st.markdown("Ejemplos clave: `age vs y` y `duration vs y`.")

        col1_7, col2_7 = st.columns(2)
        with col1_7:
            num_sel7 = st.selectbox("Variable numérica:", da.numericas,
                                    index=da.numericas.index("age") if "age" in da.numericas else 0,
                                    key="biv7_num")
        with col2_7:
            cat_sel7 = st.selectbox("Variable categórica:", da.categoricas,
                                    index=da.categoricas.index("y") if "y" in da.categoricas else 0,
                                    key="biv7_cat")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            da.plot_bivariado_num_cat(num_sel7, cat_sel7, ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.violinplot(data=df, x=cat_sel7, y=num_sel7, palette="Set3", ax=ax)
            ax.set_title(f"Violinplot — {num_sel7} por {cat_sel7}")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("#### Comparación de grupos — estadísticas")
        st.dataframe(da.comparar_grupos(num_sel7, cat_sel7), use_container_width=True)

    # ── Ítem 8 ────────────────────────────────────────────────
    with tabs[7]:
        st.markdown("### 🔗 Ítem 8 — Análisis bivariado (categórico vs categórico)")
        st.markdown("Ejemplos clave: `education vs y` y `contact vs y`.")

        col1_8, col2_8 = st.columns(2)
        with col1_8:
            cat_8a = st.selectbox("Variable 1:", da.categoricas,
                                  index=da.categoricas.index("education") if "education" in da.categoricas else 0,
                                  key="biv8a")
        with col2_8:
            cat_8b = st.selectbox("Variable 2 (referencia):", da.categoricas,
                                  index=da.categoricas.index("y") if "y" in da.categoricas else 0,
                                  key="biv8b")

        tabla = pd.crosstab(df[cat_8a], df[cat_8b], normalize="index") * 100

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            tabla.plot(kind="bar", stacked=True, colormap="tab10", ax=ax)
            ax.set_title(f"{cat_8a} vs {cat_8b} (%)")
            ax.set_ylabel("Porcentaje")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(tabla.round(1), annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title(f"Heatmap: {cat_8a} vs {cat_8b}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Ítem 9 ────────────────────────────────────────────────
    with tabs[8]:
        st.markdown("### 🎛️ Ítem 9 — Análisis basado en parámetros seleccionados")
        st.markdown("Análisis dinámico según columnas elegidas por el usuario.")

        # selectbox + multiselect + slider + checkbox siempre visibles
        col1_9, col2_9 = st.columns(2)
        with col1_9:
            col_num9 = st.selectbox("Variable numérica (eje Y):", da.numericas, key="din9_num")
            col_cat9 = st.selectbox("Variable categórica (grupos):", da.categoricas, key="din9_cat")

        with col2_9:
            cats_disp = sorted(df[col_cat9].dropna().unique().tolist())
            cats_sel9 = st.multiselect(
                "Categorías a incluir (multiselect):",
                cats_disp,
                default=cats_disp[:4] if len(cats_disp) >= 4 else cats_disp,
                key="din9_multi",
            )

        rango9 = st.slider(
            f"Rango de {col_num9}:",
            float(df[col_num9].min()),
            float(df[col_num9].max()),
            (float(df[col_num9].quantile(0.05)), float(df[col_num9].quantile(0.95))),
            key="din9_slider",
        )
        mostrar_media9 = st.checkbox("📏 Mostrar línea de media", value=True, key="din9_chk")

        if cats_sel9:
            df_fil9 = df[
                df[col_cat9].isin(cats_sel9) &
                df[col_num9].between(rango9[0], rango9[1])
            ]
            st.caption(f"Registros filtrados: {len(df_fil9):,} de {len(df):,}")

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                for cat in cats_sel9:
                    sub = df_fil9[df_fil9[col_cat9] == cat][col_num9].dropna()
                    if len(sub) > 1:
                        sns.kdeplot(sub, ax=ax, label=str(cat), fill=True, alpha=0.25)
                if mostrar_media9 and len(df_fil9) > 0:
                    media_fil = df_fil9[col_num9].mean()
                    ax.axvline(media_fil, color="red", linestyle="--",
                               label=f"Media: {media_fil:.1f}")
                ax.set_title(f"Distribución de {col_num9} por {col_cat9}")
                ax.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df_fil9, x=col_cat9, y=col_num9, palette="Set2", ax=ax,
                            order=cats_sel9)
                ax.set_title(f"Boxplot — {col_num9} por {col_cat9}")
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("#### Estadísticas por grupo (filtrado)")
            st.dataframe(
                df_fil9.groupby(col_cat9)[col_num9].agg(["count", "mean", "median", "std"]).round(2),
                use_container_width=True,
            )
        else:
            st.info("Selecciona al menos una categoría para mostrar el análisis.")

    # ── Ítem 10 ────────────────────────────────────────────────
    with tabs[9]:
        st.markdown("### 💡 Ítem 10 — Hallazgos clave del EDA")
        st.markdown("Resumen visual e insights principales derivados del análisis.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribución variable objetivo `y`")
            conteo_y = da.conteo_cat("y")
            prop_y = da.proporcion_cat("y")

            fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
            axes[0].pie(
                conteo_y.values,
                labels=[f"{i}\n({v:.1f}%)" for i, v in zip(conteo_y.index, prop_y.values)],
                colors=["#c62828", "#2e7d32"],
                startangle=90,
            )
            axes[0].set_title("Aceptación de campaña (y)")

            top_jobs = df[df["y"] == "yes"]["job"].value_counts().head(6)
            sns.barplot(x=top_jobs.values, y=top_jobs.index, palette="Greens_r", ax=axes[1])
            axes[1].set_title("Top empleos — aceptaron")
            axes[1].set_xlabel("Conteo")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("#### Correlación entre variables numéricas")
            fig, ax = plt.subplots(figsize=(7, 5))
            da.plot_heatmap_corr(da.numericas, ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### 🔍 Insights principales")
            insights = [
                ("📌 Tasa de conversión baja (~11%)",
                 "Solo 1 de cada 9 clientes aceptó la campaña. El dataset está desbalanceado."),
                ("📌 Duración del contacto — factor clave",
                 "Los clientes que aceptaron tuvieron conversaciones significativamente más largas."),
                ("📌 Jubilados y estudiantes lideran",
                 "Tienen mayor tasa de conversión proporcional, aunque no son los más contactados."),
                ("📌 Canal celular más efectivo",
                 "El contacto telefónico móvil supera al fijo en tasa de conversión."),
                ("📌 Mes de mayo — pico de contactos",
                 "La mayoría de contactos se realizó en mayo, pero no es el mes con más conversiones."),
                ("📌 Indicadores macroeconómicos",
                 "Periodos de euribor bajo coinciden con mayor aceptación de la campaña."),
            ]
            for titulo, texto in insights:
                st.markdown(f"""
                <div class="card-amarillo">
                <b>{titulo}</b><br>{texto}
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MÓDULO 4 – CONCLUSIONES (5 conclusiones, enfoque en decisiones)
# ═══════════════════════════════════════════════════════════════
elif modulo == "✅ Conclusiones":

    if st.session_state.df is None:
        st.warning("⚠️ Primero debes cargar el dataset.")
        st.stop()

    st.title("✅ Conclusiones Finales")
    st.markdown("5 conclusiones claras basadas en el EDA. Enfoque en **toma de decisiones**.")
    st.markdown("---")

    conclusiones = [
        {
            "n": "1",
            "titulo": "La campaña tiene una tasa de conversión crítica (~11%)",
            "texto": (
                "Aproximadamente el 89% de los clientes respondió 'no' a la campaña. "
                "Este desbalanceo indica que la segmentación actual es ineficiente. "
                "<b>Decisión:</b> La institución debe replantear los criterios de selección "
                "de clientes a contactar para mejorar el retorno sobre esfuerzo de campaña."
            ),
            "tipo": "card-rojo",
        },
        {
            "n": "2",
            "titulo": "La duración del contacto es el indicador de interés más fuerte",
            "texto": (
                "Los clientes que aceptaron la campaña tuvieron contactos significativamente más largos "
                "(mediana > 500 segundos vs ~100 en rechazos). "
                "<b>Decisión:</b> Capacitar a los ejecutivos para extender el diálogo con clientes "
                "potencialmente interesados podría mejorar la conversión."
            ),
            "tipo": "card-azul",
        },
        {
            "n": "3",
            "titulo": "Jubilados y estudiantes: segmentos de mayor conversión relativa",
            "texto": (
                "Aunque no son los grupos más numerosos en el dataset, muestran tasas de aceptación "
                "proporcionalmente más altas que otros perfiles laborales. "
                "<b>Decisión:</b> Focalizar campañas específicas para estos segmentos con mensajes "
                "personalizados puede incrementar la efectividad global."
            ),
            "tipo": "card-verde",
        },
        {
            "n": "4",
            "titulo": "El canal de comunicación impacta directamente los resultados",
            "texto": (
                "El contacto por celular ('cellular') supera al teléfono fijo ('telephone') "
                "en tasa de conversión. "
                "<b>Decisión:</b> La institución debería redirigir presupuesto operativo "
                "hacia el canal móvil, reduciendo el uso del canal telefónico fijo."
            ),
            "tipo": "card-azul",
        },
        {
            "n": "5",
            "titulo": "El contexto macroeconómico define ventanas de oportunidad",
            "texto": (
                "Periodos con tasas euribor bajas y mayor confianza del consumidor coinciden "
                "con mayor aceptación de productos financieros. "
                "<b>Decisión:</b> Planificar las campañas de mayor intensidad en periodos "
                "de condiciones económicas favorables aumenta la probabilidad de conversión "
                "sin incrementar el número de contactos."
            ),
            "tipo": "card-verde",
        },
    ]

    for c in conclusiones:
        st.markdown(f"""
        <div class="{c['tipo']}">
        <b>Conclusión {c['n']}: {c['titulo']}</b><br><br>
        {c['texto']}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### 🚀 Recomendaciones estratégicas para el equipo comercial")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card-azul">
        🎯 <b>Segmentar mejor</b><br>
        Enfocar contactos en jubilados, estudiantes y perfiles con historial de aceptación.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card-azul">
        📱 <b>Priorizar canal móvil</b><br>
        Redirigir esfuerzos del canal telefónico fijo al celular para mayor efectividad.
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card-azul">
        📅 <b>Timing económico</b><br>
        Intensificar campañas en periodos de euribor bajo y alta confianza del consumidor.
        </div>
        """, unsafe_allow_html=True)
