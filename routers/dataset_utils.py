"""Script que recoge funciones auxiliares relacionadas con los datasets. 
Estas funciones pueden ser modificaciones de los datos etc. y deberán ser de propósito 
general, válidas para cualquier dataset.
    """

from io import BytesIO

import pandas as pd
import numpy as np
import streamlit as st
from numpy.typing import NDArray
from streamlit.runtime.uploaded_file_manager import UploadedFile

from routers.aidtec.exceptions import WrongDatasetError


def codificar_labels(y_test:pd.DataFrame, labels_map:dict) -> np.ndarray:
    """Devuelve el dataset aplicando el labels_map

    Parameters
    ----------
    y_test : pd.DataFrame
        _description_
    labels_map : dict
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    y_test_tic = y_test.replace(labels_map)
    return y_test_tic

def y_preds_to_csv(X_test_raw:pd.DataFrame, y_preds:pd.DataFrame) -> bytes:
    """Devuelve el DataFrame original X_test_raw con las predicciones y_preds
    fusionadas en última columna

    Parameters
    ----------
    X_test_raw : pd.DataFrame
        _description_
    y_preds_raw : pd.DataFrame
        _description_
        target_col_name : str
        El nombre de la columna de y_preds

    Returns
    -------
    bytes
        Retorna el dataset Para poder descargarlo desde Streamlit
    """
    # Concatenamos los dataset
    X_download = pd.concat([X_test_raw, y_preds], axis=1)
    output = X_download.to_csv(index=False).encode('utf-8')
    return output

def get_n_outliers(row: NDArray) -> tuple[int]:
    """Función que recibe un array (o registro de un dataset) y retorna una lista con
    dos elementos: el numero de outliers por arriba
    y el num de outliers por abajo

    Parameters
    ----------
    row : np.ndarray
        _description_

    Returns
    -------
    list
        _description_
    """
    q3 = row.quantile(0.75)
    q1 = row.quantile(0.25)
    iqr = q3 - q1
    bigo_max = q3 + 1.5 * iqr
    bigo_min = q1 - 1.5 * iqr
    outliers_up = sum(row > bigo_max)
    outliers_down = sum(row < bigo_min)
    return outliers_up, outliers_down

@st.cache_data()
def load_df(df_bytes: UploadedFile) -> pd.DataFrame:
    """Carga el dataset en bytes y lo
    devuelve o lanza excepción

    Parameters
    ----------
    df_bytes : _type_
        _description_

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    WrongDatasetError
        _description_
    """
    try:
        return pd.read_csv(BytesIO(df_bytes.read()), index_col=0)
    except Exception as exc:
        raise WrongDatasetError(exc)