"""Script que recoge la lógica del desafío relacionado con el Hill vs Valley."""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Any
import os
import random
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize

from streamlit_utils import texto, imagen_con_enlace, añadir_salto
from routers.dataset_val_utils import (verificar_dataset_vacio,
                                            verificar_columnas_correctas,
                                            verificar_no_class,
                                            verificar_y_test_binario,
                                            verificar_columna_unica,
                                            verificar_valores_concretos,
                                            verificar_cantidad_registros)
from routers.metrics_utils import (plotear_matriz_confusion,
                                plot_confmat,
                                plot_roc_auc,
                                plot_precision_recall_curve,
                                computar_otras_metricas,
                                computar_accuracies
                                )
from routers.dataset_utils import codificar_labels, y_preds_to_csv, get_n_outliers
from routers.plot_utils import plotear_contorno, plotear_boxplot_outliers, plot_decision_regions

COLUMNAS_CORRECTAS = {f'V{i}' for i in range(1,101)}
VALORES_CORRECTOS = {'x', 'o', 'b'}
labels_map = {"colina": 1, "valle": 0}
inverted_labels_map = {v: k for k, v in labels_map.items()}

@st.cache_data()
def cargar_df(df_bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(df_bytes.read()))

def mostrar_resumen_modelo(model:SVC) -> None:
    """Muestra información de un modelo SVC en streamlit

    Parameters
    ----------
    model : SVC
        _description_
    """
    # Mostrar parámetros del modelo
    params: dict[str, Any] = model.get_params()
    texto("Parámetros del modelo", centrar=True, font_size=20)
    st.dataframe(params, use_container_width=True)

    # Mostrar número de vectores de soporte
    n_support_vectors:np.ndarray = model.n_support_
    dict_supp = {k: v for k,v in zip(labels_map, n_support_vectors)}
    texto("Número de vectores de soporte por clase", centrar=True, font_size=20)
    st.dataframe(dict_supp, use_container_width=True)

@st.cache_data()
def preprocess_hillvalley(df:pd.DataFrame) -> pd.DataFrame:
    """Transforma el dataframe original sin labels en 2 columnas, outliers up y outliers down
    donde cada una representa el numero de ouliers por arriba y por abajo

    Parameters
    ----------
    df : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    X = df.copy()
    # aplicamos la función auxiliar
    X_outliers = X.apply(get_n_outliers, axis=1, result_type='expand')
    X_outliers.columns = ["OutUp", "OutDown"]
    return X_outliers

def inferir(model:SVC, X_test:np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Devuelve una tupla con dataframe con y_preds y los y_probs como array

    Parameters
    ----------
    model : keras.Model
        _description_
    X_test : np.ndarray
        _description_

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        _description_
    """
    y_conf:np.ndarray = model.predict_proba(X_test)
    y_preds = model.predict(X_test)
    return pd.DataFrame(y_preds, columns=['Class']), y_conf

def hillvalley_model():
    imagen_con_enlace('https://i.imgur.com/WpRy6v7.jpg', 'https://kopuru.com/challenge/challenge-hill-valley/')
    añadir_salto()
    texto("""'Este desafío es más complejo que los demás y requiere un nivel de abstracción superior.<br>
        Pero, ¡No te preocupes! si has podido con los demás este no se te va a resistir.
            El objetivo es construir un modelo predictivo que indique si se trata de un valle o una colina, para ello cada registro representa 100 puntos en un gráfico bidimensional.
            Cuando se trazan en orden (de 1 a 100) como coordenada Y, los puntos crearán una colina (una “subida” en el terreno) o un valle (una “caída” en el terreno).<br>
            Como se trata de un ejercicio de prueba/entrenamiento, te proponemos que lo resuelvas utilizando el clasificador support vector machine y 10 fold cross validation.'""", font_size=15, formato='i')
    st.divider()
    texto("Cargar los datos", formato='b')
    añadir_salto()
    # Cargar el X_test con el uploader
    X_test_bytes = st.file_uploader("Sube el archivo **X_test** en formato csv", type=['csv'])

    if X_test_bytes is not None:
        # Instanciamos el dataset pasandolo por el método read_csv
        X_test_raw = cargar_df(X_test_bytes)
        # Verificamos que haya algo dentro
        verificar_dataset_vacio(X_test_raw)
        # Verificar que no haya columna label o target o class
        verificar_no_class(X_test_raw)
        # Verificar que el nombre de las columnas sea el esperado
        verificar_columnas_correctas(X_test_raw, COLUMNAS_CORRECTAS)
        # Verificamos que los valores sean los correcots
        #verificar_valores_concretos(X_test_raw, VALORES_CORRECTOS)
        # Preprocesamos el dataset para poder pasarlo por el modelo y lo guardamos en X_test
        try:
            X_test = preprocess_hillvalley(X_test_raw)
        except Exception as e:
            st.error(f"Se ha producido un error procesando X_test. Error: {e}")
            st.stop()
        st.success('OK')
        # Guardamos en la sesión. Guardamos también el nombre del archivo original
        if st.session_state.get("hillvalley") is None:
            st.session_state["hillvalley"] = {}
        st.session_state["hillvalley"].update({
            "X_test_raw": X_test_raw,
            "X_test": X_test,
            "X_test_filename": os.path.splitext(X_test_bytes.name)[0]
        })
        # Posibilidad de mostrar el dataframe
        if st.toggle("Visualizar **X_test**"):
            st.dataframe(X_test_raw, use_container_width=True)        

    if (X_test:=st.session_state.get("hillvalley", {}).get("X_test", None)) is not None:
        #### VISUALIZAR MODELO ####
        st.divider()
        texto("Visualizar", formato='b')
        with st.expander(f"Expande para visualizar registros de **X_test** en forma gráfica"):
            X_test_raw = st.session_state.get("hillvalley", {}).get("X_test_raw")
            max_indice = len(X_test) - 1
            indice = st.number_input("Escoge un índice y pulsa Enter", 0, max_indice)
            # Ploteamos el tablero
            plot = plotear_contorno(X_test_raw, indice)
            st.pyplot(plot)
            plot = plotear_boxplot_outliers(X_test_raw, indice)
            st.pyplot(plot)

        #### PREDECIR ####
        st.divider()
        texto("Predecir", formato='b')
        with open('models/hillvalley_svc_STM.pkl', 'rb') as f:            
            model = pickle.load(f)
        añadir_salto()
        # Mostrar detalles del modelo
        with st.expander("Ver detalles del modelo **SVC**"):
            mostrar_resumen_modelo(model)
        
        inferir_btn = st.button("Predecir")
        if inferir_btn:            
            # Guardamos en sesión y_preds y y_preds_raw
            try:
                with st.spinner("Calculando..."):
                    y_preds, y_prob = inferir(model, X_test)
                y_preds_raw = y_preds.replace(inverted_labels_map)
                st.session_state["hillvalley"].update({"y_preds": y_preds,
                                                    "y_preds_raw": y_preds_raw,
                                                    "y_prob": y_prob})
            except Exception as e:
                st.error(f"Se ha producido un error al lanzar las predicciones: {e}")
                st.stop()
            st.success("Inferencia completada correctamente.")

        if (y_preds:=st.session_state.get("hillvalley", {}).get("y_preds")) is not None:
            y_preds_raw = st.session_state.get("hillvalley", {}).get("y_preds_raw")
            X_test_raw = st.session_state.get("hillvalley", {}).get("X_test_raw")
            X_test_filename:str = st.session_state.get("hillvalley", {}).get("X_test_filename")
            # Posibilidad de visualizar y_preds
            if st.toggle("Visualizar **y_preds**"):
                st.dataframe(y_preds_raw, width=200, hide_index=False)            
            # Boton para descargar                
            st.download_button(
                label="Descargar predicciones",
                data=y_preds_to_csv(X_test_raw, y_preds, "Class"),
                file_name=X_test_filename + "_with_preds_STM.csv",
                mime='text/csv',
                help="Descarga el dataset original con las predicciones en csv",
                )
            
            #### EVALUAR ####
            st.divider()
            texto("Evaluar", formato='b')
            añadir_salto()
            y_test_bytes = st.file_uploader("Sube el archivo **y_test** en formato csv", type=['csv'])
            
            if y_test_bytes is not None:
                # Instanciamos el dataset pasandolo por el método read_csv
                y_test_raw = cargar_df(y_test_bytes)
                # Verificar que no esté vacío
                verificar_dataset_vacio(y_test_raw)
                # Verificar que solo haya una columna
                verificar_columna_unica(y_test_raw)
                # Verificar que sea binario; solo 2 tipos de valores en la columna
                verificar_y_test_binario(y_test_raw)
                # Verificar que haya el mismo numero de valores que en y_preds
                verificar_cantidad_registros(y_test_raw, y_preds_raw)
                # Verificamos que haya solo 0 y 1
                verificar_valores_concretos(y_test_raw, [0, 1])
                st.success('OK')
                # En este caso No hace falta codificar ya que y_test ya viene codificado
                y_test = y_test_raw.to_numpy()
                # Guardamos en la sesión
                st.session_state["hillvalley"].update({
                    "y_test": y_test,
                    "y_test_raw": y_test_raw})

            # Evaluación y_preds vs y_test
            if (y_test:=st.session_state.get("hillvalley", {}).get("y_test")) is not None:
                y_prob = st.session_state.get("hillvalley", {}).get("y_prob")
                y_test_raw = st.session_state.get("hillvalley", {}).get("y_test_raw")
                # Posibilidad de visualizar y_test
                if st.toggle("Visualizar **y_test**"):
                    st.dataframe(y_test_raw, width=200, hide_index=False)
                añadir_salto()
                # Añadimos accuracy y el balanced accuracy
                col1, col2 = st.columns(2)
                acc, balanced_acc = computar_accuracies(y_test, y_preds)
                with col1:
                    st.metric("Accuracy", f"{acc:.2%}")
                with col2:
                    st.metric("Balanced accuracy", f"{balanced_acc:.2%}")
                añadir_salto()
                col1, col2 = st.columns(2)
                with col1:
                    texto("Matriz de Confusión", formato='b', font_size=20)
                    plt = plot_confmat(y_test, y_preds, list(labels_map))
                    st.pyplot(plt) 
                with col2:
                    texto("ROC AUC", formato='b', font_size=20)
                    # y_prob sale con shape n_samples, n_classes. Sacamos el argmax
                    plt = plot_roc_auc(y_test, y_prob.argmax(axis=1))
                    st.pyplot(plt)
                añadir_salto()
                col1, col2 = st.columns(2)
                with col1:
                    texto("Regiones separación", formato='b', font_size=20)
                    plt = plot_decision_regions(X_test.values, 
                                                y_test.flatten(), 
                                                model, 
                                                x_label="Outliers por abajo", 
                                                y_label="Outliers por arriba",
                                                legend_list=list(labels_map))
                    st.pyplot(plt)
                with col2:
                    texto("Otras métricas", formato='b', font_size=20)
                    precision, recall, f1, mcc = computar_otras_metricas(y_test, y_preds)
                    col1, col2, = st.columns(2)
                    with col1:
                        st.metric("Precision", f"{precision:.2%}")
                        st.metric("Recall", f"{recall:.2%}")
                    with col2:
                        st.metric("F1 score", f"{f1:.2%}")
                        st.metric("MCC", f"{mcc:.2%}")

if __name__ == '__main__':
    hillvalley_model()