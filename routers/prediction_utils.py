
from typing import Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def inferir_multiclass(
        model,
        X_test: pd.DataFrame,
        col_name: str='Class',
        label_decoder: Union[dict, LabelEncoder]=None
        ) -> tuple[pd.DataFrame, NDArray[np.float_]]:
    """Corre el modelo y saca tanto las predicciones 
    como un array de probabilidades

    Parameters
    ----------
    model : _type_
        _description_
    X_test : np.ndarray
        _description_
    col_name : str, optional
        _description_, by default 'Class'
    reverse_mapping : dict, optional
        _description_, by default None

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        _description_
    """
    y_preds:NDArray[np.float_] = model.predict(X_test)

    y_prob:NDArray[np.float_] = model.predict_proba(X_test)
    if isinstance(label_decoder, dict):
        y_preds: list[NDArray[np.float_]] = [label_decoder[label] for label in y_preds]
    elif isinstance(label_decoder, LabelEncoder):
        y_preds: list[NDArray[np.float_]] = label_decoder.inverse_transform(y_preds)

    return pd.DataFrame(y_preds, columns=[col_name], index=X_test.index), y_prob