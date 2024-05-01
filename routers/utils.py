"""Script para recoger las funciones generales"""

import pickle


def deserialize(filename: str) -> object:
    """Deserializa un objeto con pickle
    y lo devuelve

    Parameters
    ----------
    filename : str
        _description_

    Returns
    -------
    object
        _description_
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj