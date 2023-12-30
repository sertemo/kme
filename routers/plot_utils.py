"""Script que recoge funciones de ploteo generales relacionadas con la visualicación de los datasets
    """
import pandas as pd
import matplotlib.pyplot as plt
from routers.dataset_utils import get_n_outliers


def plotear_contorno(X:pd.DataFrame, indice:int) -> plt.Figure:
    """Dado un registro que representa una serie de puntos de una curva
    plotea el contorno y calcula la media el máximo y el mínimo.

    Parameters
    ----------
    X : pd.DataFrame
        _description_
    indice : int
        _description_

    Returns
    -------
    plt.Figure
        _description_
    """
    n_columnas = len(X.columns)
    valores_y = X.iloc[indice, :]
    media = valores_y.mean()
    maximo = valores_y.max()
    minimo = valores_y.min()
    plt.figure(figsize=(14, 4))
    plt.plot(range(n_columnas), valores_y)
    plt.axhline(y = media, color = 'r', linestyle = '--')
    plt.axhline(y = maximo, color = 'g', linestyle = ':')
    plt.axhline(y = minimo, color = 'g', linestyle = ':')
    plt.title(f"Muestra: {indice} - Max: {maximo}, Media: {media:.2f}, Min: {minimo}")
    plt.ylabel("Coordenada Y")
    return plt

def plotear_boxplot_outliers(X:pd.DataFrame, indice:int) -> plt.Figure:
    """Dado un dataframe y un índice, plotea un boxplot y calcula el número
    de outliers por arriba y por abajo

    Parameters
    ----------
    X : pd.DataFrame
        _description_
    indice : int
        _description_

    Returns
    -------
    plt.Figure
        _description_
    """
    row = X.iloc[indice]
    plt.figure(figsize=(14, 4))
    plt.boxplot(row)
    plt.tight_layout()
    outliers_arriba, outliers_abajo = get_n_outliers(row)
    titulo = f"Outliers arriba: {outliers_arriba}" if outliers_arriba else f"Outliers abajo: {outliers_abajo}"
    plt.title(titulo)
    plt.xticks((1,), [f"Índice: {indice}"])
    return plt