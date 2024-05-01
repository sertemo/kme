"""Script con funciones de ploteo"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier
import streamlit as st


def plotear_distribucion_variables(df: pd.DataFrame, n_cols: int=4) -> plt.Figure:
    """Plotea las distribuciones de todas las variables

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    n_cols : int, optional
        _description_, by default 4

    Returns
    -------
    plt.Figure
        _description_
    """
    df_float_cols = df.select_dtypes('float64')

    # Número total de columnas en el DataFrame
    total_columns = df_float_cols.shape[1]

    # Calcular número de filas de gráficos necesarias
    num_plot_rows = (total_columns + n_cols - 1) // n_cols
    # Crear una figura y un arreglo de ejes

    fig, axes = plt.subplots(
        nrows=num_plot_rows,
        ncols=n_cols,
        figsize=(n_cols * 4, num_plot_rows * 3)
        )
    # Aplanar el arreglo de ejes para facilitar la iteración
    axes = axes.flatten()

    fig.suptitle('Distribución de los valores de todas las variables del modelo')

    # Iterar sobre las columnas del DataFrame y los ejes
    for ax, column in zip(axes, df_float_cols.columns):
        sns.histplot(df_float_cols[column], ax=ax, kde=True)
        ax.grid(True)

    # Esconder ejes extras si los hay
    for i in range(total_columns, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return plt

@st.cache_resource()
def plot_rf_trees(_rf: RandomForestClassifier, num_tree: int):
    """Plotea un determinado arbol de decisión
    de un RandomForest

    Parameters
    ----------
    _rf : RandomForestClassifier
        _description_
    num_tree : int
        _description_

    Returns
    -------
    _type_
        _description_
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    tree = _rf.estimators_[num_tree]
    plot_tree(
        tree,
        filled=True,
        ax=ax,
        max_depth=2,
        class_names=[str(c) for c in _rf.classes_],
        feature_names=_rf.feature_names_in_
        )
    ax.set_title(f"Árbol de Decisión {num_tree}")
    return fig

def plotear_correlaciones(df: pd.DataFrame) -> plt.Figure:
    plt.figure(figsize=(14, 8))
    sns.heatmap(df.corr(numeric_only=True),
            annot=True, linewidth=.5,
            cmap='viridis')
    return plt