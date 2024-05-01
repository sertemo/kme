"""Script que recoge la lógica del desafío de AidTec Solutions respecto a la calidad del vino"""


import io
import os
from textwrap import dedent

from numpy.typing import NDArray
import pandas as pd
import pickle
import requests
from sklearn.preprocessing import LabelEncoder
import streamlit as st


from routers.dataset_val_utils import (verificar_dataset_vacio,
                                            verificar_columnas_correctas,
                                            verificar_no_class,
                                            verificar_cantidad_registros,
                                            verificar_columna_unica,
                                            verificar_valores_concretos)
from routers.metrics_utils import (plot_confmat,
                                plot_roc_auc_multiclass,
                                plot_precision_recall_curve,
                                computar_otras_metricas,
                                computar_accuracies
                                )
from routers.dataset_utils import codificar_labels, y_preds_to_csv, load_df
from routers.prediction_utils import inferir_multiclass
from routers.utils import deserialize
from routers.aidtec.exceptions import WrongDatasetError
from routers.aidtec.transformers import WineDatasetTransformer
from routers.aidtec.models import SerializableClassifier
from routers.aidtec.plot import plot_rf_trees, plotear_correlaciones
from streamlit_utils import texto, imagen_con_enlace, añadir_salto


NOMBRE_DESAFIO = 'wine'
TARGET_COL_NAME = 'calidad'
COLUMNAS_CORRECTAS = {'acidez fija', 'acidez volatil', 'acido citrico', 'azucar residual',
                    'densidad', 'dioxido de azufre total', 'dioxido de azufre libre', 'cloruros',
                    'pH', 'sulfatos', 'alcohol', 'color', 'year', 'calidad'}


@st.cache_resource()
def load_model_from_disk(model_path: str) -> SerializableClassifier:
    model = SerializableClassifier.load(model_path=model_path)
    return model

@st.cache_resource(show_spinner="Descargando el modelo ...")
def load_model_from_url(url: str) -> SerializableClassifier:
    r = requests.get(url)
    if r.status_code == 200:
        model = pickle.load(io.BytesIO(r.content))
        return model


@st.cache_data()
def preprocess_wine(X_test_raw:pd.DataFrame) -> pd.DataFrame:
    """Dado un dataset original, preprocesa las variables de la misma forma
    en la que han sido procesadas en el training

    Parameters
    ----------
    X_test_raw : pd.DataFrame
        _description_
    wine_transformer : WineDatasetTransformer
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # Instanciamos el winetransformer
    wine_transformer = WineDatasetTransformer(
    shuffle=False,
    corregir_alcohol=True,
    corregir_densidad=True,
    color_interactions=False,
    remove_outliers=False,
    standardize=False,
    densidad_alcohol_interaction=True,
    ratio_diox=True,
    rbf_diox=True,
    drop_columns=['color', 'densidad', 'alcohol', 'year', 'dioxido de azufre libre'],
    )

    X_test_transformed: pd.DataFrame = wine_transformer.fit_transform(X_test_raw)
    return X_test_transformed.drop(columns=[TARGET_COL_NAME])


def aidtec_model():
    imagen_con_enlace('https://i.imgur.com/vAnLXOY.jpg', 
                    'https://kopuru.com/challenge/modelo-de-prediccion-de-calidad-en-el-vino-para-aidtec-solutions/')
    añadir_salto()
    texto(
        dedent(
    """AidTec surge en 2023 como una Startup Universitaria con un firme propósito: desarrollar soluciones que optimicen las labores del campo, haciéndolo más sostenible para sus productores. Como parte de sus proyectos de investigación sobre la calidad del terreno, y cómo esta afecta al producto final, el vino, se embarcan en un estudio de la predicción de calidades del vino, basado en algunos parámetros que se miden en laboratorio.
    \nEsta información es parte de un proyecto mayor que buscará la relación entre la calidad del vino y su correlación con los datos del terreno y la planta.\n    
    \nConoce a AidTec Solutions y su labor en el mundo del campo, donde apuestan por la sostenibilidad, a través de su Web y su LinkedIn.
    \nAcompáñanos en este reto en el que, mediante un modelo predictivo, se busca obtener el algoritmo con mayor porcentaje de acierto, capaz de predecir la nota de calidad que obtendrá cada vino."""
        ),
    font_size=15,
    formato='i'
    )
    st.divider()
    texto("Cargar los datos", formato='b')
    añadir_salto()
    # Cargar el X_test con el uploader
    X_test_bytes_wine = st.file_uploader("Sube el archivo **X_test** en formato csv", type=['csv'])

    if X_test_bytes_wine is not None:
        # Instanciamos el dataset pasandolo por el método read_csv
        try:
            X_test_raw = load_df(X_test_bytes_wine)
            st.dataframe(X_test_raw)
        except WrongDatasetError as exc:
            st.error(f"Se ha producido un error al cargar el archivo: {exc}")
            st.stop()
        # Verificamos que haya algo dentro
        verificar_dataset_vacio(X_test_raw)
        # Verificar que no haya columna label o target o class
        # Verificar que el nombre de las columnas sea el esperado
        verificar_columnas_correctas(X_test_raw, COLUMNAS_CORRECTAS)
        # Verificamos que los valores sean los correcots
        # Preprocesamos el dataset para poder pasarlo por el modelo y lo guardamos en X_test
        try:
            X_test = preprocess_wine(X_test_raw)
        except Exception as e:
            st.error(f"Se ha producido un error procesando **X_test**. Revisa el dataset. Si el error persiste contacta con tejedor.moreno@gmail.com. Error: {e}")
            st.stop()
        st.success('OK')
        # Guardamos en la sesión. Guardamos también el nombre del archivo original
        if st.session_state.get(NOMBRE_DESAFIO) is None:
            st.session_state[NOMBRE_DESAFIO] = {}
        st.session_state[NOMBRE_DESAFIO].update({
            "X_test_raw": X_test_raw,
            "X_test": X_test,
            "X_test_filename": os.path.splitext(X_test_bytes_wine.name)[0]
        })
        # Posibilidad de mostrar el dataframe
        if st.toggle("Visualizar **X_test**"):
            st.dataframe(X_test_raw, use_container_width=True)        

    # Modelo
    if (X_test:=st.session_state.get(NOMBRE_DESAFIO, {}).get("X_test", None)) is not None:
        st.divider()
        texto("Visualizar", formato='b')
        with st.expander(f"Expande para visualizar las distribuciones de las variables de **X_test**"):
            X_test_raw = st.session_state.get(NOMBRE_DESAFIO, {}).get("X_test_raw")
            # Ploteamos el map de calor
            plot = plotear_correlaciones(X_test_raw)
            st.pyplot(plot)

        st.divider()
        texto("Predecir", formato='b')
        #model = load_model_from_disk('models/wine_random_forest_STM.pkl')
        model = load_model_from_url(
            'https://github.com/sertemo/kme/raw/main/models/wine_random_forest_STM.pkl'
            )
        if model is not None:
            st.success("Modelo cargado correctamente!")
        else:
            st.error("Fallo al cargar el modelo.")
            st.stop()
        
        añadir_salto()
        # Mostrar detalles del modelo
        with st.expander("Ver detalles del modelo **RandomForest**"):
            img1 = plot_rf_trees(model.classifier, 0)
            texto("Primera rama", formato='b', font_size=20, centrar=True)
            st.pyplot(img1)
            img2 = plot_rf_trees(model.classifier, 1)
            texto("Segunda rama", formato='b', font_size=20, centrar=True)
            st.pyplot(img2)
            texto("(...)", formato='b', font_size=20, centrar=True)

        inferir_btn = st.button("Predecir")
        if inferir_btn:            
            # Guardamos en sesión y_preds y y_preds_raw
            with st.spinner("Calculando..."):
                try:
                    label_decoder = deserialize('models/wine_label_encoder.pkl')
                    y_preds, y_prob = inferir_multiclass(model, X_test, TARGET_COL_NAME, label_decoder)
                    st.session_state[NOMBRE_DESAFIO].update({"y_preds": y_preds,
                                                    "y_prob": y_prob})
                except Exception as e:
                    st.error(f"Se ha producido un error al lanzar las predicciones: {e}")
                    st.stop()
            st.success("Inferencia completada correctamente.")

        if (y_preds:=st.session_state.get(NOMBRE_DESAFIO, {}).get("y_preds")) is not None:
            X_test_raw: pd.DataFrame = st.session_state.get(NOMBRE_DESAFIO, {}).get("X_test_raw")
            X_test_filename:str = st.session_state.get(NOMBRE_DESAFIO, {}).get("X_test_filename")
            # Posibilidad de visualizar y_preds
            if st.toggle("Visualizar **y_preds**"):
                st.dataframe(y_preds, width=200, hide_index=False)            
            # Boton para descargar
            st.download_button(
                label="Descargar predicciones",
                data=y_preds_to_csv(X_test_raw.drop(columns=['calidad']), y_preds),
                file_name= X_test_filename + '_STM_preds' + '.csv',
                mime='text/csv',
                help="Descarga el dataset original con las predicciones en csv",
                )
            st.divider()

            texto("Evaluar", formato='b')
            añadir_salto()
            y_test_bytes = st.file_uploader("Sube el archivo **y_test** en formato csv", type=['csv'])
            
            if y_test_bytes is not None:
                # Instanciamos el dataset pasandolo por el método read_csv
                y_test_raw = load_df(y_test_bytes)
                # Verificar que no esté vacío
                verificar_dataset_vacio(y_test_raw)
                # Verificar que solo haya una columna
                verificar_columna_unica(y_test_raw)
                # Verificar que sea binario; solo 2 tipos de valores en la columna
                # Verificar que haya el mismo numero de valores que en y_preds
                verificar_cantidad_registros(y_test_raw, y_preds)
                st.success('OK')
                # Guardamos en la sesión
                st.session_state[NOMBRE_DESAFIO].update({"y_test_raw": y_test_raw})

            # Evaluación y_preds vs y_test
            if (y_test_raw := st.session_state.get(NOMBRE_DESAFIO, {}).get("y_test_raw")) is not None:
                y_prob = st.session_state.get(NOMBRE_DESAFIO, {}).get("y_prob")
                # Posibilidad de visualizar y_test
                if st.toggle("Visualizar **y_test**"):
                    st.dataframe(y_test_raw, width=200, hide_index=False)
                añadir_salto()
                # Añadimos accuracy y el balanced accuracy
                col1, col2 = st.columns(2)
                acc, balanced_acc = computar_accuracies(y_test_raw, y_preds)
                with col1:
                    st.metric("Accuracy", f"{acc:.2%}")
                with col2:
                    st.metric("Balanced accuracy", f"{balanced_acc:.2%}")
                añadir_salto()
                col1, col2 = st.columns(2)
                with col1:
                    texto("Matriz de Confusión", formato='b', font_size=20)
                    plt = plot_confmat(y_test_raw, y_preds, [str(i) for i in range(3, 10)])
                    st.pyplot(plt) 
                with col2:
                    texto("ROC AUC", formato='b', font_size=20)
                    plt = plot_roc_auc_multiclass(
                        y_test_raw,
                        y_prob,
                        dict(enumerate([str(i) for i in range(3, 10)],
                                    start=3))
                        )
                    st.pyplot(plt)
                añadir_salto(2)
                col1, col2, col3 = st.columns(3)
                with col2:
                    texto("Otras métricas", formato='b', font_size=20)
                    precision, recall, f1, mcc = computar_otras_metricas(y_test_raw, y_preds, [str(i) for i in range(3, 10)])
                    col1, col2,  = st.columns(2)
                    with col1:
                        st.metric("Precision", f"{precision:.2%}")
                        st.metric("Recall", f"{recall:.2%}")
                    with col2:
                        st.metric("F1 score", f"{f1:.2%}")
                        st.metric("MCC", f"{mcc:.2%}")

if __name__ == '__main__':
    aidtec_model()