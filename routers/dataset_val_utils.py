"""Script para recoger todas las validaciones y verificaciones relacionadas con los datasets.
Serán validaciones generales aplicables a cualquier tipo de dataset
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Iterable, Union

@st.cache_data()
def verificar_dataset_vacio(X_test:pd.DataFrame) -> Union[bool, None]:
    """Verifica que el dataframe tenga registros

    Parameters
    ----------
    X_test : pd.DataFrame
        _description_
    Returns
    _______
    bool : Retorna False si no está vacio
    """
    if (len(X_test) == 0):
        error_msg = f"El dataframe subido está vacío."
        st.error(error_msg)
        st.stop()
    return False

@st.cache_data()
def verificar_no_class(X_test:pd.DataFrame, nombres_targets:list=['target', 'label', 'class']) -> Union[bool, None]:
    """Verifica que el dataset no tenga variable Targets
    Se le puede pasar el nombre de las columnas a excluir. Por defecto: target, label y class
    Pasar los nombres siempre en MINÚSCULAS.
    Devuelve True si efectivamente no hay columna clase
    """
    for columna in X_test.columns:
        if columna.lower() in nombres_targets:
            error_msg = f"La columna **{columna}** no está permitida. Es necesario pasar X_test sin los targets"
            st.error(error_msg)
            st.stop()
    return True

@st.cache_data()
def verificar_columnas_correctas(X_test:pd.DataFrame, columnas_correctas:Iterable) -> Union[bool, None]:
    """Verifica si el nombre de las columnas de un dataset son las correctas y se corresponden
    con las pasadas por columnas_correctas

    Parameters
    ----------
    X_test : pd.DataFrame
        _description_
    columnas_correctas : Iterable
        _description_

    Returns
    -------
    _type_
        Devuelve True si son correctas
    """
    for columna in X_test.columns:
        if columna not in columnas_correctas:
            st.error(f"**{columna}** no es una columna correcta. Los nombres correctos son: **{', '.join(columnas_correctas)}**")
            st.stop()
    return True


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

def verificar_cantidad_registros(y:pd.DataFrame, y_comparacion:pd.DataFrame) -> None:
    """Verifica que el número de registros de y sea el mismo que el de
    y_comparacion. Si no lo es muestar un error

    Parameters
    ----------
    y : pd.DataFrame
        _description_
    y_comparacion : pd.DataFrame
        _description_
    """
    longitud_y = len(y)
    longitud_comparacion = len(y_comparacion)
    if not longitud_y == len(y_comparacion):
        st.error(f"El número de registros no es correcto. Se esperaban {longitud_comparacion} se han encontrado {longitud_y}")
        st.stop()