
import joblib
import pickle
from pathlib import Path
from typing import cast, Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class SerializableMixin:
    """Clase que implementa el método save
    para serializar un modelo"""

    def save(self, model_path: Path) -> None:
        """Serializa el modelo usando joblib"""
        with open(model_path, "wb") as f:
            pickle.dump(self, f)


class DeserializableMixin(SerializableMixin):
    """Clase para cargar y deserializar
    un modelo guardado

    Parameters
    ----------
    SerializableMixin : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    @classmethod
    def load(cls, model_path: str) -> "DeserializableMixin":
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
        return classifier


class SerializableClassifier(
    BaseEstimator,
    ClassifierMixin,
    DeserializableMixin,
):
    """Wrapper de un clasificador para
    que sea serializable y deserializable
    con joblib

    Parameters
    ----------
    BaseEstimator : _type_
        _description_
    ClassifierMixin : _type_
        _description_
    DeserializableMixin : _type_
        _description_
    """

    def __init__(self, classifier: BaseEstimator) -> None:
        self.classifier: BaseEstimator = classifier

    def fit(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: NDArray[np.float64] | pd.DataFrame,
    ) -> "SerializableClassifier":
        self.classifier.fit(X, y)
        return self

    def predict(self, X: NDArray[np.float64] | pd.DataFrame) -> NDArray[np.int64]:
        check_is_fitted(self.classifier)
        predictions: NDArray[np.int64] = self.classifier.predict(X)
        return predictions

    def __getattr__(self, attr: Any) -> Any:
        """Delega atributos al clasificador subyacente si no se encuentran en 'self'."""
        return getattr(self.classifier, attr)