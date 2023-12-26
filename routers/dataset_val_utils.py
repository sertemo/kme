"""Script para recoger todas las validaciones y verificaciones relacionadas con los datasets.
Serán validaciones generales aplicables a cualquier tipo de dataset
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Iterable

def verificar_dataset_vacio(X_test:pd.DataFrame) -> None:
    """Verifica que el dataframe tenga registros

    Parameters
    ----------
    X_test : pd.DataFrame
        _description_
    """
    if (len(X_test) == 0):
        st.error(f"El dataframe subido está vacío.")
        st.stop()

def verificar_no_class(X_test:pd.DataFrame, nombres_targets:list=['target', 'label', 'class']) -> None:
    """Verifica que el dataset no tenga variable Targets
    Se le puede pasar el nombre de las columnas a excluir. Por defecto: target, label y class
    Pasar los nombres siempre en MINÚSCULAS
    """
    for columna in X_test.columns:
        if columna.lower() in nombres_targets:
            st.error(f"La columna **{columna}** no está permitida. Es necesario pasar X_test sin los targets")
            st.stop()
            break

def verificar_columnas_correctas(X_test:pd.DataFrame, columnas_correctas:Iterable) -> None:
    for columna in X_test.columns:
        if columna not in columnas_correctas:
            st.error(f"**{columna}** no es una columna correcta. Los nombres correctos son: {', '.join(columnas_correctas)}")

def verificar_columna_unica(y_test:pd.DataFrame) -> None:
    """Verifica que haya una sola columna en el dataframe,
    Si no lanza error

    Parameters
    ----------
    y_test : pd.DataFrame
        _description_
    """
    if len(y_test.columns) > 1:
        st.error("El dataframe **y_test** solo puede tener 1 columna")
        st.stop()
    

def verificar_y_test_binario(y_test:pd.DataFrame) -> None:
    """Verifica que en la primera columna (y se supone que única) solo haya 2 valores distintos.
    Da igual cuales. 
    Usar en problemas de clasificación binarios

    Parameters
    ----------
    y_test : pd.DataFrame
        _description_
    col_target_name : str, optional
        _description_, by default "Class"
    """
    valores_unicos:np.ndarray = y_test[y_test.columns[0]].unique()
    if (num_valores_unicos:=y_test.nunique().values[0]) != 2:
        st.error(f"El dataframe no es correcto: solo puede haber 2 tipos de valores diferentes y hay {num_valores_unicos}: **{', '.join(valores_unicos)}**")
        st.stop()

def verificar_valores_concretos(X_test:pd.DataFrame, valores_correctos:Iterable) -> None:
    """Verifica que todos los valores del dataset sean unos valores concretos predefinidos

    Parameters
    ----------
    X_test : pd.DataFrame
        _description_
    valores_correctos : Iterable
        _description_
    """
    # TODO: Sacar los valores distintos para mostrarlos
    # Verificar si todos los elementos en el DataFrame están en los valores permitidos
    if not X_test.map(lambda x: x in valores_correctos).all().all():
        st.error(f"Hay valores erróneos en el dataset. Los valores correctos son: **{', '.join(valores_correctos)}**")
        st.stop()