"""Script que recoge la lógica del desafío relacionado con el Tic Tac Toe."""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from tensorflow import keras
from typing import Union
import time
import os
import matplotlib.pyplot as plt
from matplotlib import colors

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
from routers.dataset_utils import codificar_labels, y_preds_to_csv

COLUMNAS_CORRECTAS = {'top-left-square', 'top-middle-square', 'top-right-square',
    'middle-left-square', 'middle-middle-square', 'middle-right-square',
    'bottom-left-square', 'bottom-middle-square', 'bottom-right-square'}
VALORES_CORRECTOS = {'x', 'o', 'b'}
to_colors = {'o': 150,
            'x': 30,
            'b': 255}
inverted_to_color = {v: k for k, v in to_colors.items()}
labels_map = {"positive": 1, "negative": 0}
inverted_labels_map = {v: k for k, v in labels_map.items()}

@st.cache_resource()
def plot_jugada(X:pd.DataFrame, indice:int, fontsize:int=50) -> plt.Figure:
    """Devuelve el plot del tablera de 3 en raya con la jugada del índice pasado
    X es dataframe sin labels

    Parameters
    ----------
    X : pd.DataFrame
        _description_
    y : pd.Series
        _description_
    indice : int
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    cmap = colors.ListedColormap(['black', 'gray', 'white'])
    bounds=[0, 80, 200, 300]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    if isinstance(X, pd.DataFrame):
        if 'Class' in X.columns:
            raise ValueError("Hay que pasar X. Elimina la columna 'Class'")
        jugada_colors = X.iloc[indice, :].map(to_colors).values.reshape(3, 3)

    elif isinstance(X, np.ndarray):
        jugada_colors = X[indice]
        if jugada_colors.shape != (3, 3):
            jugada_colors = jugada_colors.reshape(3, 3)
    plt.imshow(jugada_colors, cmap=cmap, norm=norm)
    plt.axis(True)
    plt.xticks(np.arange(-0.5, 2.5), range(3))
    plt.yticks(np.arange(-0.5, 2.5), range(3))
    plt.grid(color='#D3A121', linestyle='--', linewidth=1.5)
    plt.title(f'Índice: {indice}')
    for i, _ in enumerate(jugada_colors):
        for j, q in enumerate(jugada_colors[i]):
            plt.text(j-0.2, i, inverted_to_color[q], color='w', fontsize=fontsize)
    return plt

def load_df(df_bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(df_bytes.read()), dtype=str)

def mostrar_resumen_modelo(model:keras.Model) -> None:
    # Captura la salida de model.summary()
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    st.markdown('```' + summary_string + '```')

@ st.cache_data()
def preprocess_tictactoe(X:pd.DataFrame, y:pd.Series=None) -> Union[tuple[np.ndarray, pd.Series], tuple[np.ndarray]]:
    # Creamos copias
    X_tic = X.copy()
    # Reemplazamos con ints
    X_tic = (X_tic.replace(to_colors) / 255.).values.reshape((-1, 3, 3, 1))
    if y is not None:
        y_tic = y.copy()
        # One Shoteamos los labels
        y_tic = y_tic.replace(labels_map)
        return X_tic, y_tic
    return X_tic

def inferir(model:keras.Model, X_test:np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
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
    y_conf:np.ndarray = model.predict(X_test)
    y_preds = (y_conf >= 0.5).astype('int')
    return pd.DataFrame(y_preds, columns=['Class']), y_conf

def tictactoe_model():
    imagen_con_enlace('https://i.imgur.com/ErNyvlS.jpg', 'https://kopuru.com/challenge/challenge-tic-tac-toe/')
    añadir_salto()
    texto("""'En este reto, vamos a jugar (y ganar) al típico 3 en raya. ¿Te apuntas?
            Esta base de datos codifica el conjunto completo de posibles configuraciones del tablero, al final de los juegos de tres en raya, donde suponemos que “x” ha jugado primero. 
            El objetivo de este desafío es “ganar por x”.
            Como se trata de un ejercicio de prueba/entrenamiento, te proponemos que lo resuelvas utilizando 
            el clasificador XGBoost “extreme gradient boosting” (refuerzo de gradientes extremos) y 10 fold cross validation. 
            Pero te invitamos a que expandas tu creatividad y nos sorprendas.'""", font_size=15, formato='i')
    st.divider()
    texto("Cargar los datos", formato='b')
    añadir_salto()
    # Cargar el X_test con el uploader
    X_test_bytes_tictactoe = st.file_uploader("Sube el archivo **X_test** en formato csv", type=['csv'])

    if X_test_bytes_tictactoe is not None:
        # Instanciamos el dataset pasandolo por el método read_csv
        X_test_raw = load_df(X_test_bytes_tictactoe)
        # Verificamos que haya algo dentro
        verificar_dataset_vacio(X_test_raw)
        # Verificar que no haya columna label o target o class
        verificar_no_class(X_test_raw)
        # Verificar que el nombre de las columnas sea el esperado
        verificar_columnas_correctas(X_test_raw, COLUMNAS_CORRECTAS)
        # Verificamos que los valores sean los correcots
        verificar_valores_concretos(X_test_raw, VALORES_CORRECTOS)
        # Preprocesamos el dataset para poder pasarlo por el modelo y lo guardamos en X_test
        try:
            X_test = preprocess_tictactoe(X_test_raw)
        except Exception as e:
            st.error(f"Se ha producido un error procesando X_test. Error: {e}")
            st.stop()
        st.success('OK')
        # Guardamos en la sesión. Guardamos también el nombre del archivo original
        if st.session_state.get("tictactoe") is None:
            st.session_state["tictactoe"] = {}
        st.session_state["tictactoe"].update({
            "X_test_raw": X_test_raw,
            "X_test": X_test,
            "X_test_filename": os.path.splitext(X_test_bytes_tictactoe.name)[0]
        })
        # Posibilidad de mostrar el dataframe
        if st.toggle("Visualizar **X_test**"):
            st.dataframe(X_test_raw, use_container_width=True)        

    # Modelo
    if (X_test:=st.session_state.get("tictactoe", {}).get("X_test", None)) is not None:
        st.divider()
        texto("Visualizar", formato='b')
        with st.expander(f"Expande para visualizar registros de **X_test**"):
            X_test_raw = st.session_state.get("tictactoe", {}).get("X_test_raw")
            indice = st.number_input("Escoge un índice y pulsa Enter", 0, len(X_test) - 1)
            # Ploteamos el tablero
            plot = plot_jugada(X_test_raw, indice)
            st.pyplot(plot)

        st.divider()
        texto("Predecir", formato='b')
        model = keras.models.load_model('models/tictactoe_convnet_STM.model')
        añadir_salto()
        # Mostrar detalles del modelo
        with st.expander("Ver detalles del modelo **Convnet**"):
            mostrar_resumen_modelo(model)

        inferir_btn = st.button("Predecir")
        if inferir_btn:            
            # Guardamos en sesión y_preds y y_preds_raw
            try:
                with st.spinner("Calculando..."):
                    y_preds, y_prob = inferir(model, X_test)
                y_preds_raw = y_preds.replace(inverted_labels_map)
                st.session_state["tictactoe"].update({"y_preds": y_preds,
                                                    "y_preds_raw": y_preds_raw,
                                                    "y_prob": y_prob})
            except Exception as e:
                st.error(f"Se ha producido un error al lanzar las predicciones: {e}")
                st.stop()
            st.success("Inferencia completada correctamente.")

        if (y_preds:=st.session_state.get("tictactoe", {}).get("y_preds")) is not None:
            y_preds_raw = st.session_state.get("tictactoe", {}).get("y_preds_raw")
            X_test_raw = st.session_state.get("tictactoe", {}).get("X_test_raw")
            X_test_filename:str = st.session_state.get("tictactoe", {}).get("X_test_filename")
            # Posibilidad de visualizar y_preds
            if st.toggle("Visualizar **y_preds**"):
                st.dataframe(y_preds_raw, width=200, hide_index=False)            
            # Boton para descargar                
            st.download_button(
                label="Descargar predicciones",
                data=y_preds_to_csv(X_test_raw, y_preds_raw),
                file_name=X_test_filename + "_with_preds_STM.csv",
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
                verificar_y_test_binario(y_test_raw)
                # Verificar que haya el mismo numero de valores que en y_preds
                verificar_cantidad_registros(y_test_raw, y_preds_raw)
                st.success('OK')
                # Transformar y_test, pasarlo a 0 y 1. Label Encoder
                y_test = codificar_labels(y_test_raw, labels_map)
                # Guardamos en la sesión
                st.session_state["tictactoe"].update({
                    "y_test": y_test,
                    "y_test_raw": y_test_raw})

            # Evaluación y_preds vs y_test
            if (y_test:=st.session_state.get("tictactoe", {}).get("y_test")) is not None:
                y_prob = st.session_state.get("tictactoe", {}).get("y_prob")
                y_test_raw = st.session_state.get("tictactoe", {}).get("y_test_raw")
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
                    plt = plot_roc_auc(y_test, y_prob)
                    st.pyplot(plt)

                col1, col2 = st.columns(2)
                with col1:
                    texto("Curva Precision-Recall", formato='b', font_size=20)
                    plt = plot_precision_recall_curve(y_test, y_prob)
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
    tictactoe_model()