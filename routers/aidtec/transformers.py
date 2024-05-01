
"""Script que recoge los transformers personalizados"""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
    OneHotEncoder,
)

from routers.aidtec.exceptions import WrongColumnName, WrongColumnType


class WineDatasetTransformer(TransformerMixin, BaseEstimator):
    """Transformer específico del proyecto AidTec"""

    def __init__(
        self,
        corregir_alcohol: bool = True,
        corregir_densidad: bool = True,
        color_interactions: bool = True,
        densidad_alcohol_interaction: bool = True,
        ratio_diox: bool = True,
        rbf_diox: bool = True,
        remove_outliers: bool = False,
        standardize: bool = False,
        log_transformation: list[str] | None = None,
        drop_columns: list[str] | None = None,
        shuffle: bool = True,
    ) -> None:
        """Inicializa los parámetros de transformación
        a aplicar"""
        self.corregir_alcohol = corregir_alcohol
        self.corregir_densidad = corregir_densidad
        self.color_interactions = color_interactions
        self.densidad_alcohol_interaction = densidad_alcohol_interaction
        self.ratio_diox = ratio_diox
        self.rbf_diox = rbf_diox
        self.sf_coords = 27, 126  # Coordenadas de los clusters
        self.isolation_forest = IsolationForest(random_state=42)
        self.gamma_1 = 0.003
        self.gamma_2 = 2.5e-4
        self.remove_outliers = remove_outliers
        self.standardize = standardize
        self.sc = StandardScaler()
        self.oh_encoder = OneHotEncoder(drop="if_binary", sparse_output=False)
        self.log_transformation_list = log_transformation
        self.drop_columns_list = drop_columns
        self.shuffle = shuffle

    def _filtrar_alcohol_malos(self, feature: pd.Series) -> pd.Series:
        """Devuelve los valores filtrados de
        la variable alcohol

        Parameters
        ----------
        feature : pd.Series
            _description_

        Returns
        -------
        pd.Series
            Devuelve los valores erróneos de la variable alcohol
        """
        return feature[feature.apply(len) > 5]

    def _corregir_valores_alcohol(self, feature: pd.Series) -> pd.Series:
        """Corrige los valores de la variable alcohol
        moviendo la coma hacia la izquierda de los valores
        filtrados erróneos.

        Parameters
        ----------
        feature : pd.Series
            _description_

        Returns
        -------
        pd.Series
            Devuelve la variable corregida

        Raises
        ------
        ValueError
            _description_
        """
        # Copiamos la feature
        feature_ = feature.copy()
        # Filtramos los valores erroneos
        feature_malos = self._filtrar_alcohol_malos(feature_)

        # Quitamos los puntos
        if feature_malos.dtype != "O":
            raise ValueError("La feature debe ser de tipo object")

        feature_limpio = feature_malos.str.replace(".", "", regex=False)
        # Si empieza por 8 o 9 agregamos un 0 delante
        feature_limpio[feature_limpio.str.startswith(("8", "9"))] = "0" + feature_limpio
        # Añadimos el punto desde la posición 2
        feature_limpio = (
            feature_limpio.str.slice(0, 2) + "." + feature_limpio.str.slice(1, 4)
        )
        # Modificamos en el dataset completo
        feature_[feature_malos.index] = feature_limpio
        return feature_.astype("float64")  # Devolvemos el tipo

    def _corregir_valores_densidad(self, feature: pd.Series) -> pd.Series:
        """Corrige los valores erróneos de la variable
        densidad dividiendo por 10 hasta que se llegue a la
        unidad.

        Parameters
        ----------
        feature : pd.Series
            _description_

        Returns
        -------
        pd.Series
            Devuelve la variable corregida
        """
        # Copiamos la feature
        feature_ = feature.copy()
        # Filtramos los valores erroneos
        feature_malos = feature_[feature_ > 2]

        # Creamos función que procese los valores malos
        def dividir_por_diez(valor: float) -> float:
            while valor >= 10:  # Mientras el valor sea igual o mayor a 10,
                valor /= 10  # dividir entre 10.
            return valor

        feature_corregidos = feature_malos.apply(dividir_por_diez)

        # Sustituimos los corregidos
        feature_[feature_malos.index] = feature_corregidos

        return feature_

    def fit(
        self, X: NDArray[np.float_] | pd.DataFrame, y=None
    ) -> "WineDatasetTransformer":
        coord1, coord2 = self.sf_coords
        self.rbf_transformer_1 = FunctionTransformer(
            rbf_kernel, kw_args=dict(Y=[[coord1]], gamma=self.gamma_1)
        )
        self.rbf_transformer_2 = FunctionTransformer(
            rbf_kernel, kw_args=dict(Y=[[coord2]], gamma=self.gamma_2)
        )
        # Validación de logs
        if self.log_transformation_list is not None:
            for col in self.log_transformation_list:
                if col not in X:
                    raise WrongColumnName(f"La columna {col} no es correcta")

        # Validacion drops
        if self.drop_columns_list is not None:
            for col in self.drop_columns_list:
                if col not in X:
                    raise WrongColumnName(f"La columna {col} no es correcta")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_ = X.copy()
        if self.corregir_alcohol:
            # Corregimos alcohol
            X_["alcohol"] = self._corregir_valores_alcohol(X_["alcohol"])
        if self.corregir_densidad:
            # Corregimos densidad
            X_["densidad"] = self._corregir_valores_densidad(X_["densidad"])

        # Binarizamos la variable color
        X_["color"] = self.oh_encoder.fit_transform(X_[["color"]]).astype("int64")

        if self.color_interactions:
            # Interacciones con la variable color
            X_["color_acidez_vol"] = X_["color"] * X_["acidez volatil"]
            X_["color_dioxido_azufre"] = X_["color"] * X_["dioxido de azufre total"]
            X_["color_cloruros"] = X_["color"] * X_["cloruros"]
        if self.densidad_alcohol_interaction:
            # Interaccion densidad alcohol
            # Hay que verificar que se pueda multiplicar
            if X_["alcohol"].dtype == "object":
                raise WrongColumnType(
                    "Seguramente tengas que corregir la variable alcohol"
                )
            X_["densidad_alcohol"] = X_["densidad"] * X_["alcohol"]
        if self.ratio_diox:
            X_["SO2_l / SO2_tot"] = (
                X_["dioxido de azufre libre"] / X_["dioxido de azufre total"]
            )
            # Creamos variables distancias a los modos de diox azufre total
        if self.rbf_diox:
            diox_simil_1 = self.rbf_transformer_1.transform(
                X_[["dioxido de azufre total"]]
            )
            diox_simil_2 = self.rbf_transformer_2.transform(
                X_[["dioxido de azufre total"]]
            )
            X_["diox_simil_1"] = diox_simil_1
            X_["diox_simil_2"] = diox_simil_2
        if self.remove_outliers:
            # Hay que asegurarse de haber corregido alcohol antes
            # Sino da error
            if X_["alcohol"].dtype == "object":
                raise WrongColumnType(
                    "Seguramente tengas que corregir la variable alcohol"
                )
            self.outlier_pred = self.isolation_forest.fit_predict(X_)
            X_ = X_.iloc[self.outlier_pred == 1, :].reset_index(drop=True)

        if self.standardize:
            # Hay que verificar si hay columnas tipo objeto
            X_ = pd.concat(
                [
                    pd.DataFrame(
                        self.sc.fit_transform(X_.select_dtypes("float64")),
                        columns=self.sc.feature_names_in_,
                        index=X_.index,
                    ),
                    X_.select_dtypes("object"),
                    X_.select_dtypes("int64"),
                ],
                axis=1,
            )

        if self.log_transformation_list is not None:
            for col in self.log_transformation_list:
                X_[col] = X_[col].apply(np.log)

        if self.drop_columns_list is not None:
            X_ = X_.drop(columns=self.drop_columns_list)

        if self.shuffle:
            X_ = X_.sample(len(X_), random_state=42)

        return X_

    def get_feature_names_out(self, names=None):
        super().get_feature_names_out()