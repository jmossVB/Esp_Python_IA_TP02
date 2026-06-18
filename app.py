"""
Caso de Estudio N°1 – BankMarketing EDA
Especialización Python for Analytics | DMC Institute
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="BankMarketing EDA", layout="wide")


# ─────────────────────────────────────────────
# CLASE PRINCIPAL  (POO)
# ─────────────────────────────────────────────
class DataAnalyzer:
    """Encapsula estadísticas descriptivas, clasificación de variables
    y helpers de visualización para un DataFrame de Pandas."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def classify_variables(self):
        num = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        return num, cat

    def descriptive_stats(self, cols=None):
        df = self.df[cols] if cols else self.df
        return df.describe(include="all").T

    def missing_summary(self):
        total = self.df.isnull().sum()
        percent = (total / len(self.df) * 100).round(2)
        return pd.DataFrame({"Nulos": total, "Porcentaje (%)": percent}) \
            .sort_values("Nulos", ascending=False)

    def fig_distribution(self, col, bins=30):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(self.df[col].dropna(), bins=bins, edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de {col}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def fig_boxplot(self, num_col, cat_col):
        order = self.df.groupby(cat_col)[num_col].median() \
            .sort_values(ascending=False).index
        data_plot = [self.df.loc[self.df[cat_col] == v, num_col].dropna()
                     for v in order]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(data_plot, labels=order)
        ax.set_xlabel(cat_col)
        ax.set_ylabel(num_col)
        ax.set_title(f"{num_col} por {cat_col}")
        plt.xticks(rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def fig_barplot(self, col, top_n=10, horizontal=False):
        vc = self.df[col].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(7, 4))
        if horizontal:
            ax.barh(vc.index[::-1], vc.values[::-1])
            ax.set_xlabel("Frecuencia")
        else:
            ax.bar(vc.index, vc.values)
            ax.set_ylabel("Frecuencia")
            plt.xticks(rotation=35, ha="right")
        ax.set_title(f"Distribución de {col}")
        ax.grid(axis="x" if horizontal else "y", alpha=0.3)
        fig.tight_layout()
        return fig

    def fig_crosstab_pct(self, col1, col2):
        ct = pd.crosstab(self.df[col1], self.df[col2], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(7, 4))
        ct.plot(kind="bar", ax=ax)
        ax.set_xlabel(col1)
        ax.set_ylabel("Porcentaje (%)")
        ax.set_title(f"{col1} vs {col2} (% por fila)")
        ax.legend(title=col2)
        plt.xticks(rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def fig_correlation(self, cols=None):
        num_cols = cols or self.classify_variables()[0]
        corr = self.df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Matriz de correlación (numéricas)")
        fig.tight_layout()
        return fig


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file, sep=";")


def render_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def get_df():
    df = st.session_state.get("df", None)
    if df is None:
        st.warning("⚠️ Primero carga el dataset en la sección 'Carga de Datos'.")
        st.stop()
    return df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("BankMarketing EDA")
    menu = st.radio(
        "Navegar",
        ["Home",
         "Carga de Datos",
         "EDA - Información",
         "EDA - Distribuciones",
         "EDA - Bivariado",
         "EDA - Análisis Dinámico",
         "Conclusiones"],
    )
    st.markdown("---")
    st.write("Dataset: BankMarketing.csv")


# ─────────────────────────────────────────────
# MÓDULO 1 – HOME
# ─────────────────────────────────────────────
if menu == "Home":
    st.title("Bank Marketing EDA")
    st.subheader("Análisis Exploratorio de Datos — Campaña Bancaria")

    st.markdown("### Objetivo del análisis")
    st.write("""
    Una institución financiera registró una caída en la efectividad de sus
    campañas de marketing: de 12% a 8% en los últimos 6 meses.
    Este proyecto aplica un EDA sobre los datos de la última campaña para
    identificar patrones, relaciones entre variables y perfiles de clientes
    con mayor propensión a aceptar el producto ofrecido.

    El análisis no construye modelos predictivos; el foco está en comprender
    los datos y extraer insights accionables que orienten decisiones comerciales.
    """)

    st.markdown("### Sobre el dataset")
    st.write("""
    BankMarketing.csv contiene 41,188 registros de clientes contactados en
    una campaña de depósitos a plazo fijo. Incluye variables demográficas
    (edad, empleo, educación), financieras (créditos, mora) y de la campaña
    (canal, duración, intentos). La variable objetivo `y` indica si el cliente
    suscribió el depósito.
    """)

    st.markdown("### Autor")
    st.write("""
    - **Nombre:** [Tu nombre completo]
    - **Curso:** Especialización Python for Analytics
    - **Institución:** DMC Institute
    - **Año:** 2025
    """)

    st.markdown("### Tecnologías utilizadas")
    st.write("Python, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Programación Orientada a Objetos")


# ─────────────────────────────────────────────
# MÓDULO 2 – CARGA
# ─────────────────────────────────────────────
elif menu == "Carga de Datos":
    st.title("Carga del Dataset")

    uploaded = st.file_uploader(
        "Sube el archivo BankMarketing.csv (separador: punto y coma)",
        type=["csv"],
    )

    if uploaded is None:
        st.info("Por favor, carga el archivo CSV para continuar.")
        st.stop()

    df = load_data(uploaded)
    st.session_state["df"] = df

    st.success(f"Archivo cargado correctamente: {uploaded.name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", f"{df.shape[0]:,}")
    col2.metric("Columnas", f"{df.shape[1]}")
    col3.metric("Tamaño estimado",
                f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    st.markdown("### Vista previa (primeras 5 filas)")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Tipos de datos")
    dtype_df = pd.DataFrame({"Tipo": df.dtypes.astype(str),
                              "Nulos": df.isnull().sum()})
    st.dataframe(dtype_df, use_container_width=True)


# ─────────────────────────────────────────────
# MÓDULO 3 – EDA INFORMACIÓN
# ─────────────────────────────────────────────
elif menu == "EDA - Información":
    df = get_df()
    analyzer = DataAnalyzer(df)
    st.title("EDA — Información General")

    tabs = st.tabs(["Ítem 1: Dataset Info", "Ítem 2: Variables",
                    "Ítem 3: Estadísticas", "Ítem 4: Valores Faltantes"])

    # ÍTEM 1
    with tabs[0]:
        st.header("Ítem 1 — Información general del dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", f"{df.shape[0]:,}")
        col2.metric("Columnas", f"{df.shape[1]}")
        col3.metric("Valores nulos totales", int(df.isnull().sum().sum()))

        st.markdown("#### Tipos de datos por columna")
        buf = io.StringIO()
        df.info(buf=buf)
        st.code(buf.getvalue(), language="text")

        st.write("""
        **Discusión:** el dataset no presenta valores nulos. Todas las 21
        variables están completamente pobladas, lo que facilita el análisis
        sin necesidad de imputación previa.
        """)

    # ÍTEM 2
    with tabs[1]:
        st.header("Ítem 2 — Clasificación de variables")
        num_cols, cat_cols = analyzer.classify_variables()

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Variables numéricas ({len(num_cols)}):**")
            st.write(num_cols)
        with col2:
            st.write(f"**Variables categóricas ({len(cat_cols)}):**")
            st.write(cat_cols)

        resumen = pd.DataFrame({
            "Variable": df.columns,
            "Tipo pandas": df.dtypes.astype(str).values,
            "Categoría": ["Numérica" if c in num_cols else "Categórica"
                          for c in df.columns],
            "Únicos": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(resumen, use_container_width=True, hide_index=True)

    # ÍTEM 3
    with tabs[2]:
        st.header("Ítem 3 — Estadísticas descriptivas")
        num_cols, _ = analyzer.classify_variables()
        show_cols = st.multiselect(
            "Selecciona variables numéricas a describir",
            options=num_cols, default=num_cols[:6],
        )
        if show_cols:
            stats = df[show_cols].describe().T.round(2)
            stats["mediana"] = df[show_cols].median().round(2)
            stats["moda"] = df[show_cols].mode().iloc[0].round(2)
            st.dataframe(stats, use_container_width=True)

            st.write(f"""
            **Interpretación:** la variable `duration` tiene media de
            {df['duration'].mean():.0f}s pero mediana de
            {df['duration'].median():.0f}s, lo cual indica sesgo a la derecha
            por llamadas muy largas. `age` se concentra alrededor de los 40 años.
            """)

    # ÍTEM 4
    with tabs[3]:
        st.header("Ítem 4 — Análisis de valores faltantes")
        miss = analyzer.missing_summary()
        st.dataframe(miss, use_container_width=True)

        total_miss = miss["Nulos"].sum()
        if total_miss == 0:
            st.success("El dataset no presenta valores nulos explícitos (NaN).")
            st.write("""
            **Discusión:** la ausencia de nulos puede indicar que el dataset ya
            fue preprocesado, o que los valores desconocidos se codificaron
            como categorías explícitas (ej. "unknown").
            """)

            _, cat_cols = analyzer.classify_variables()
            unk = {c: (df[c] == "unknown").sum()
                   for c in cat_cols if "unknown" in df[c].values}
            if unk:
                st.markdown("#### Valores 'unknown' por variable categórica")
                unk_df = pd.DataFrame.from_dict(
                    unk, orient="index", columns=["Conteo 'unknown'"])
                unk_df["% del total"] = (unk_df["Conteo 'unknown'"] / len(df) * 100).round(2)
                st.dataframe(unk_df, use_container_width=True)


# ─────────────────────────────────────────────
# MÓDULO 4 – DISTRIBUCIONES
# ─────────────────────────────────────────────
elif menu == "EDA - Distribuciones":
    df = get_df()
    analyzer = DataAnalyzer(df)
    st.title("EDA — Distribuciones")

    tabs = st.tabs(["Ítem 5: Numéricas", "Ítem 6: Categóricas"])

    # ÍTEM 5
    with tabs[0]:
        st.header("Ítem 5 — Distribución de variables numéricas")
        num_cols, _ = analyzer.classify_variables()

        sel_num = st.selectbox("Variable numérica", num_cols, index=0)
        bins = st.slider("Número de bins", 10, 80, 30)

        render_fig(analyzer.fig_distribution(sel_num, bins=bins))

        data = df[sel_num].dropna()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Media", f"{data.mean():.2f}")
        col2.metric("Mediana", f"{data.median():.2f}")
        col3.metric("Desv. Std", f"{data.std():.2f}")
        col4.metric("Sesgo", f"{data.skew():.2f}")

        st.markdown("#### Panel completo de variables numéricas")
        fig2, axes = plt.subplots(2, 5, figsize=(14, 5))
        for ax, col in zip(axes.flat, num_cols):
            ax.hist(df[col].dropna(), bins=25, edgecolor="white")
            ax.set_title(col, fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(labelsize=7)
        fig2.tight_layout()
        render_fig(fig2)

        st.write("""
        **Interpretación visual:** `duration` y `campaign` presentan fuerte
        sesgo positivo: la mayoría de los contactos son breves y pocos. Las
        variables macroeconómicas (`euribor3m`, `emp.var.rate`) muestran
        distribuciones bimodales asociadas a ciclos económicos.
        """)

    # ÍTEM 6
    with tabs[1]:
        st.header("Ítem 6 — Análisis de variables categóricas")
        _, cat_cols = analyzer.classify_variables()

        sel_cat = st.selectbox("Variable categórica", cat_cols,
                               index=cat_cols.index("job") if "job" in cat_cols else 0)
        top_n = st.slider("Top N categorías", 5, 20, 10)

        col1, col2 = st.columns(2)
        with col1:
            render_fig(analyzer.fig_barplot(sel_cat, top_n=top_n, horizontal=True))
        with col2:
            vc = df[sel_cat].value_counts().head(top_n)
            pct = (vc / len(df) * 100).round(2)
            st.dataframe(pd.DataFrame({"Frecuencia": vc, "%": pct}),
                         use_container_width=True)

        st.markdown("#### Variable objetivo `y` (tasa de conversión)")
        y_vc = df["y"].value_counts()
        fig_p, ax_p = plt.subplots(figsize=(4, 4))
        ax_p.pie(y_vc.values,
                 labels=[f"{l} ({v:,}, {v/len(df)*100:.1f}%)"
                         for l, v in zip(y_vc.index, y_vc.values)],
                 startangle=90)
        ax_p.set_title("Distribución de y")
        render_fig(fig_p)

        st.write(f"""
        **Discusión:** solo el {y_vc['yes']/len(df)*100:.1f}% de los clientes
        aceptó la campaña, lo que confirma el desbalanceo de clases. El análisis
        debe enfocarse en caracterizar ese segmento minoritario.
        """)


# ─────────────────────────────────────────────
# MÓDULO 5 – BIVARIADO
# ─────────────────────────────────────────────
elif menu == "EDA - Bivariado":
    df = get_df()
    analyzer = DataAnalyzer(df)
    st.title("EDA — Análisis Bivariado")

    tabs = st.tabs(["Ítem 7: Numérico vs y",
                    "Ítem 8: Categórico vs y",
                    "Extra: Correlaciones"])

    # ÍTEM 7
    with tabs[0]:
        st.header("Ítem 7 — Numérico vs Categórico (y)")
        num_cols, _ = analyzer.classify_variables()
        sel = st.selectbox("Variable numérica", num_cols,
                           index=num_cols.index("duration") if "duration" in num_cols else 0)

        col1, col2 = st.columns(2)
        with col1:
            render_fig(analyzer.fig_boxplot(sel, "y"))
        with col2:
            grp = df.groupby("y")[sel].agg(["mean", "median", "std"]).round(2)
            grp.columns = ["Media", "Mediana", "Desv. Std"]
            st.dataframe(grp, use_container_width=True)
            diff_mean = grp.loc["yes", "Media"] - grp.loc["no", "Media"]
            st.metric(f"Diferencia de medias (yes - no) en {sel}", f"{diff_mean:+.2f}")

        st.markdown("#### Distribución de edad por resultado (y)")
        fig_age, ax_age = plt.subplots(figsize=(7, 3.5))
        for val in ["yes", "no"]:
            ax_age.hist(df.loc[df["y"] == val, "age"], bins=30,
                        alpha=0.6, label=val, density=True)
        ax_age.set_xlabel("Edad")
        ax_age.set_ylabel("Densidad")
        ax_age.set_title("Edad según resultado de campaña")
        ax_age.legend()
        ax_age.grid(axis="y", alpha=0.3)
        fig_age.tight_layout()
        render_fig(fig_age)

        st.write("""
        **Interpretación:** los clientes que aceptaron (`yes`) tienden a tener
        mayor duración de llamada. Clientes en los extremos de edad (jóvenes
        menores de 25 y mayores de 60) muestran mayor tasa de conversión.
        """)

    # ÍTEM 8
    with tabs[1]:
        st.header("Ítem 8 — Categórico vs Categórico (y)")
        _, cat_cols = analyzer.classify_variables()
        cat_no_y = [c for c in cat_cols if c != "y"]
        sel_cat = st.selectbox("Variable categórica", cat_no_y,
                               index=cat_no_y.index("education") if "education" in cat_no_y else 0)

        col1, col2 = st.columns([3, 2])
        with col1:
            render_fig(analyzer.fig_crosstab_pct(sel_cat, "y"))
        with col2:
            ct_abs = pd.crosstab(df[sel_cat], df["y"])
            ct_pct = (ct_abs.div(ct_abs.sum(axis=1), axis=0) * 100).round(2)
            ct_pct.columns = ["no (%)", "yes (%)"]
            st.dataframe(ct_pct.sort_values("yes (%)", ascending=False),
                         use_container_width=True)

        st.write(f"""
        **Discusión:** en `{sel_cat}` se observan diferencias claras en la tasa
        de conversión entre categorías. Segmentar la campaña por este atributo
        podría mejorar la efectividad.
        """)

    # EXTRA: CORRELACIONES
    with tabs[2]:
        st.header("Matriz de correlación")
        num_cols, _ = analyzer.classify_variables()
        sel_corr = st.multiselect("Columnas a correlacionar", num_cols, default=num_cols)
        if len(sel_corr) >= 2:
            render_fig(analyzer.fig_correlation(sel_corr))
            st.write("""
            **Interpretación:** `euribor3m`, `emp.var.rate` y `nr.employed`
            tienen alta correlación entre sí (contexto macroeconómico). Usar
            solo una de ellas en modelos evitaría problemas de multicolinealidad.
            """)


# ─────────────────────────────────────────────
# MÓDULO 6 – ANÁLISIS DINÁMICO
# ─────────────────────────────────────────────
elif menu == "EDA - Análisis Dinámico":
    df = get_df()
    analyzer = DataAnalyzer(df)
    st.title("EDA — Análisis Dinámico (Ítem 9)")

    st.write("Configura el análisis seleccionando las variables de interés.")

    num_cols, cat_cols = analyzer.classify_variables()
    cat_no_y = [c for c in cat_cols if c != "y"]

    col1, col2, col3 = st.columns(3)
    with col1:
        eje_x = st.selectbox("Variable X (categórica o numérica)", cat_no_y + num_cols, index=0)
    with col2:
        eje_y = st.selectbox("Variable Y (numérica o resultado)", num_cols + ["y"], index=0)
    with col3:
        filtro_y = st.multiselect("Filtrar por resultado (y)", ["yes", "no"], default=["yes", "no"])

    min_age, max_age = int(df["age"].min()), int(df["age"].max())
    age_range = st.slider("Rango de edad", min_age, max_age, (min_age, max_age))
    mostrar_tabla = st.checkbox("Mostrar tabla de datos agrupados", value=False)

    df_f = df[(df["y"].isin(filtro_y)) &
              (df["age"] >= age_range[0]) &
              (df["age"] <= age_range[1])]

    st.write(f"Registros filtrados: **{len(df_f):,}** de {len(df):,}")

    if df_f.empty:
        st.warning("Sin datos para los filtros seleccionados.")
    else:
        if eje_x in num_cols and eje_y in num_cols:
            fig_sc, ax_sc = plt.subplots(figsize=(7, 4))
            for val in filtro_y:
                sub = df_f[df_f["y"] == val]
                ax_sc.scatter(sub[eje_x], sub[eje_y], alpha=0.3, s=10, label=val)
            ax_sc.set_xlabel(eje_x)
            ax_sc.set_ylabel(eje_y)
            ax_sc.set_title(f"{eje_x} vs {eje_y}")
            ax_sc.legend(title="y")
            ax_sc.grid(alpha=0.3)
            fig_sc.tight_layout()
            render_fig(fig_sc)
        elif eje_x in cat_no_y:
            if eje_y == "y":
                render_fig(analyzer.fig_crosstab_pct(eje_x, "y"))
            else:
                render_fig(analyzer.fig_boxplot(eje_y, eje_x))

        if mostrar_tabla and eje_x in cat_no_y and eje_y in num_cols:
            tbl = df_f.groupby(eje_x)[eje_y].agg(["mean", "median", "count"]).round(2)
            tbl.columns = ["Media", "Mediana", "N"]
            tbl = tbl.sort_values("Media", ascending=False)
            st.dataframe(tbl, use_container_width=True)


# ─────────────────────────────────────────────
# MÓDULO 7 – CONCLUSIONES
# ─────────────────────────────────────────────
elif menu == "Conclusiones":
    df = get_df()
    st.title("Hallazgos Clave y Conclusiones (Ítem 10)")

    total = len(df)
    yes_rate = df["y"].value_counts(normalize=True)["yes"] * 100
    avg_dur_yes = df.loc[df["y"] == "yes", "duration"].mean()
    avg_dur_no = df.loc[df["y"] == "no", "duration"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total registros", f"{total:,}")
    col2.metric("Tasa conversión", f"{yes_rate:.1f}%")
    col3.metric("Duración media YES", f"{avg_dur_yes:.0f}s")
    col4.metric("Duración media NO", f"{avg_dur_no:.0f}s")

    st.markdown("### Visualización resumen de hallazgos")

    fig_sum, axes = plt.subplots(1, 3, figsize=(14, 4))

    job_rate = (df[df["y"] == "yes"].groupby("job").size() /
                df.groupby("job").size() * 100).sort_values(ascending=False)
    axes[0].barh(job_rate.index, job_rate.values)
    axes[0].set_title("Tasa de conversión por empleo (%)", fontsize=9)
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].tick_params(labelsize=7)

    month_order = ["jan", "feb", "mar", "apr", "may", "jun",
                   "jul", "aug", "sep", "oct", "nov", "dec"]
    mo = df[df["month"].isin(month_order)]
    month_rate = (mo[mo["y"] == "yes"].groupby("month").size() /
                  mo.groupby("month").size() * 100).reindex(month_order, fill_value=0)
    axes[1].bar(month_rate.index, month_rate.values)
    axes[1].set_title("Tasa de conversión por mes (%)", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)

    pout_rate = (df[df["y"] == "yes"].groupby("poutcome").size() /
                 df.groupby("poutcome").size() * 100).sort_values(ascending=False)
    axes[2].bar(pout_rate.index, pout_rate.values)
    axes[2].set_title("Tasa de conversión por resultado anterior (%)", fontsize=9)
    axes[2].grid(axis="y", alpha=0.3)

    fig_sum.tight_layout()
    render_fig(fig_sum)

    st.markdown("### Conclusiones finales")

    conclusiones = [
        ("1. La duración del contacto es el predictor más relevante.",
         "Los clientes que aceptaron la campaña tuvieron llamadas significativamente "
         f"más largas (media {avg_dur_yes:.0f}s vs {avg_dur_no:.0f}s). Aunque la "
         "duración no se conoce antes de la llamada, indica que mantener "
         "conversaciones productivas es clave para la conversión."),
        ("2. Los meses de menor actividad macroeconómica son más efectivos.",
         "Marzo, septiembre, octubre y diciembre muestran las tasas de conversión "
         "más altas. Esto coincide con periodos de euribor3m bajo, donde los "
         "depósitos a plazo resultan más atractivos para el cliente."),
        ("3. El resultado de la campaña anterior es altamente predictivo.",
         "Los clientes con poutcome = success convierten a una tasa muy superior "
         "al resto. Priorizar la recontactación de clientes con historial positivo "
         "puede mejorar directamente la efectividad de la campaña."),
        ("4. El perfil de empleo influye significativamente en la conversión.",
         "Los clientes con empleos de mayor cualificación (admin., management, "
         "retired) presentan tasas de conversión más altas. Segmentar las "
         "campañas por tipo de empleo permite asignar recursos comerciales con "
         "mayor precisión."),
        ("5. La alta concentración de contactos en mayo reduce la efectividad global.",
         f"El {(df['month'] == 'may').mean()*100:.1f}% de los registros corresponden "
         "a mayo, pero con una tasa de conversión por debajo de la media. "
         "Redistribuir el esfuerzo hacia meses de mayor efectividad histórica "
         "(marzo, septiembre, octubre) podría recuperar puntos porcentuales perdidos."),
    ]

    for titulo, texto in conclusiones:
        st.markdown(f"**{titulo}**")
        st.write(texto)
        st.markdown("---")
