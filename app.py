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
