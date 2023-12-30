"""Script que recoge funciones de ploteo generales relacionadas con la visualicación de los datasets
    """
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from routers.dataset_utils import get_n_outliers
import numpy as np


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

def plot_decision_regions(X, y, 
                        classifier, test_idx=None, 
                        resolution=0.02, x_label:str="", 
                        y_label:str="", legend_list:list=[]) -> plt.Figure:
    """Plotea las regiones de separación creadas por el classifier para problemas de clasificación en 2D.
    Plotea la curva o linea creada por el clasificador para dividir las clases.
    Para problemas de clasificación multiple o binaria.

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    classifier : _type_
        _description_
    test_idx : _type_, optional
        _description_, by default None
    resolution : float, optional
        _description_, by default 0.02
    x_label : str, optional
        _description_, by default ""
    y_label : str, optional
        _description_, by default ""
        legend_list : list, by default []
        Los valores que quieres que aparezcan en la leyenda de cada clase, por ejemplo 'valle' y 'colina'
    """
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    legend_handlers = []

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                        np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.title("Separación de regiones")

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        fig = plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
        legend_handlers.append(fig)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend_handlers, legend_list, loc='best')

    # test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='Validation set')
    return plt