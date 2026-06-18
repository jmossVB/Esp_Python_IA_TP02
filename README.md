# 🏦 Bank Marketing EDA — Análisis Exploratorio de Datos

## 📋 Descripción del proyecto

Aplicación interactiva construida con **Python y Streamlit** para el análisis exploratorio del dataset **BankMarketing.csv**, correspondiente a una institución financiera que busca entender los factores que influyen en la aceptación de sus campañas de marketing.

Durante los últimos 6 meses, la efectividad cayó del **12% al 8%**, afectando los bonos de los ejecutivos comerciales. Este proyecto analiza los datos de la última campaña para descubrir relaciones y comportamientos relevantes entre las variables.

---

## 📁 Estructura del proyecto

```
bank-marketing-eda/
│
├── app.py                # Aplicación principal Streamlit
├── BankMarketing.csv     # Dataset (institución financiera)
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Este archivo
```

---

## 🔍 Módulos de la aplicación

| Módulo | Descripción |
|---|---|
| 🏠 **Home** | Presentación del proyecto y datos del autor |
| 📂 **Carga del Dataset** | Upload del CSV, validación, head y dimensiones |
| 🔍 **Análisis EDA** | 10 ítems de análisis exploratorio con visualizaciones |
| ✅ **Conclusiones** | 5 conclusiones basadas en el análisis |

### 10 ítems del EDA
1. Información general del dataset (`.info()`, tipos, nulos)
2. Clasificación de variables (función personalizada en clase POO)
3. Estadísticas descriptivas (`.describe()`, media, mediana, moda)
4. Análisis de valores faltantes (conteo y discusión)
5. Distribución de variables numéricas (histogramas + boxplots)
6. Análisis de variables categóricas (barras + proporciones)
7. Análisis bivariado numérico vs categórico (boxplot + violinplot)
8. Análisis bivariado categórico vs categórico (heatmap + barras apiladas)
9. Análisis dinámico con selectbox/multiselect/slider/checkbox
10. Hallazgos clave (insights principales del EDA)

---

## 🛠️ Tecnologías utilizadas

- **Python 3.x**
- **Pandas** — Manipulación de datos
- **NumPy** — Cálculos numéricos
- **Matplotlib** — Visualizaciones base
- **Seaborn** — Visualizaciones estadísticas
- **Streamlit** — Interfaz web interactiva

## 🧱 Programación Orientada a Objetos

Se implementó la clase `DataAnalyzer` que encapsula:
- Estadísticas descriptivas
- Clasificación de variables (función personalizada)
- Funciones de visualización y análisis

---

## 🌐 Aplicación desplegada

🔗 [Ver app en Streamlit Cloud](https://esppythoniatp02.streamlit.app/)

---

## 📊 Dataset

**BankMarketing.csv** — Dataset de campaña de marketing de una institución financiera.
- ~4,500 registros | 21 variables
- Variable objetivo: `y` (aceptó la campaña: yes/no)

---

*Proyecto desarrollado como parte de la Especialización en Python for Analytics — 2026*
