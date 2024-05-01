"""Script que recoge la lógica del desafío relacionado con la prediction del tráfico
en un polígono industrial del Pais Vasco."""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Union
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from datetime import datetime
import xgboost as xgb

from streamlit_utils import texto, imagen_con_enlace, añadir_salto
from routers.dataset_val_utils import (verificar_dataset_vacio,
                                            verificar_columnas_correctas,
                                            verificar_no_class,
                                            verificar_cantidad_registros,
                                            verificar_columna_unica,
                                            verificar_valores_concretos)
from routers.metrics_utils import (plotear_matriz_confusion,
                                plot_confmat,
                                plot_roc_auc_multiclass,
                                plot_precision_recall_curve,
                                computar_otras_metricas,
                                computar_accuracies
                                )
from routers.dataset_utils import codificar_labels, y_preds_to_csv

YEAR = 2023
MONTH = 10
COLUMNAS_CORRECTAS = {'Time', 'Date', 'Day of the week', 'CarCount',
                    'BikeCount', 'BusCount', 'TruckCount', 'Total'}
VEHICULOS = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']
DIAS_SEMANA = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FILENAME_CSV = "TraficoDataEvaluado_SERGIO TEJEDOR_v02.csv"
labels_map = {"low": 0, "normal": 1, "high": 2, "heavy": 3}
day_map = {dia: numero for dia, numero in zip(DIAS_SEMANA, range(1,8))}
inverted_labels_map = {v: k for k, v in labels_map.items()}

def plot_densidad_trafico(X_test_raw:pd.DataFrame) -> plt.Figure:
    new_df = X_test_raw.copy()
    # Renombramos algunas columnas
    new_df.rename(
        columns={"Day of the week": "Day",
                "Total": "TotalCount"},
                inplace=True)    
    # Pasamos a datetime la columna Time
    new_df.Time = pd.to_datetime(new_df.Time, errors="coerce", format="%I:%M:%S %p").dt.time
    # Creamos columna DateTime
    new_df["DateTime"] = new_df.apply(convert_to_datetime, axis=1)
    # ordenamos cronologicamente y ponemos esta columna como indice
    new_df = new_df.sort_values(by=["DateTime"]).reset_index(drop=True)
    new_df.set_index("DateTime", inplace=True)
    # Creamos la columna Daynumber con los numeros de semana siendo numeros
    new_df['DayNumber'] = new_df.Day.map(day_map)
    new_df['Hour'] = new_df.index.hour
    # Calculamos la media de tráfico por cada combinación de día de la semana y hora del día
    traffic_heatmap_data = new_df.groupby(['DayNumber', 'Hour']).mean(numeric_only=True)['TotalCount'].unstack()
    # Generamos el mapa de calor
    plt.figure(figsize=(10, 8));
    sns.heatmap(traffic_heatmap_data, cmap='viridis', linewidths=.5, annot=True, fmt=".0f", annot_kws={"size":8});
    plt.title('Mapa de Calor de Densidad de Tráfico medio por Hora y Día de la Semana')
    plt.xlabel('Hora del Día');
    plt.ylabel('Día de la Semana');
    plt.yticks(np.arange(7), DIAS_SEMANA, rotation=0)
    return plt

def mostrar_resumen_modelo(model:xgb.XGBClassifier) -> None:
    # Mostrar parámetros del modelo
    params = model.get_params()
    #st.write("Parámetros del modelo:")
    #st.json(params)
    # Mostrar la estructura del modelo (árboles)
    booster = model.get_booster()
    tree_dump = booster.get_dump()
    st.write("Estructura del modelo (primer árbol):")
    st.code(tree_dump[0])  # Muestra solo el primer árbol

@st.cache_resource()
def load_model(model_path:str) -> xgb.XGBClassifier:
    with open(model_path, 'rb') as f:            
            model = pickle.load(f)
    return model

@st.cache_data()
def load_df(df_bytes) -> pd.DataFrame:
    """recibe el objeto UploadedFile de Streamlit
    y los convierte en dataframe

    Parameters
    ----------
    df_bytes : _type_
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    return pd.read_csv(BytesIO(df_bytes.read()))

def plot_arbol_decision(model:xgb.XGBClassifier, num_arbol:int=0) -> plt.Figure:
    xgb.plot_tree(model, num_trees=num_arbol)
    return plt

def convert_to_datetime(row) -> datetime:
    date_str = f"{YEAR}-{MONTH:02d}-{row['Date']:02d} {row['Time']}"
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

@st.cache_data()
def preprocess_traffic(df:pd.DataFrame) -> pd.DataFrame:
    """Dado un dataset original, preprocesa las variables de la misma forma
    en la que han sido procesadas en el training

    Parameters
    ----------
    df : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        Dataset con los valores procesados
    """    
    new_df = df.copy()
    # Renombramos algunas columnas
    new_df.rename(
        columns={"Day of the week": "Day",
                "Total": "TotalCount"},
        inplace=True)
    # Transformamos a 24h la columna Time
    new_df.Time = pd.to_datetime(new_df.Time, errors="coerce", format="%I:%M:%S %p").dt.time    
    # Creamos columna DateTime
    new_df["DateTime"] = new_df.apply(convert_to_datetime, axis=1)
        # La nombramos como índice
    new_df.set_index("DateTime", inplace=True)
    # Añadimos la columna Hora
    new_df['Hour'] = new_df.index.hour
    # Añadimos los minutos
    new_df["Minute"] = new_df.index.minute
    # Añadimos la columna DayNumber
    new_df['DayNumber'] = new_df.Day.map(day_map)
    # Quitamos Day y Time
    new_df.drop(["Day", "Time"], axis=1, inplace=True)
    # Renombramos Date por Day
    new_df.rename(columns={"Date": "Day"}, inplace=True)
    # Reordenamos en día de la semana, número de dia del mes, hora, minuto
    new_df = new_df[["DayNumber", "Day", "Hour", "Minute", *VEHICULOS, "TotalCount"]]
    return new_df

def inferir_multiclass(model, X_test:np.ndarray, col_name:str='Class', reverse_mapping:dict=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Corre el modelo y saca tanto las predicciones como un array de probabilidades

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
    y_preds:np.ndarray = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    if reverse_mapping is not None:
        y_preds:list = [reverse_mapping[label] for label in y_preds]
    return pd.DataFrame(y_preds, columns=[col_name]), y_prob

def traffic_model():
    imagen_con_enlace('https://i.imgur.com/ghd2KVc.jpg', 
                    'https://kopuru.com/challenge/prediccion-del-trafico-a-la-entrada-del-poligono-industrial-pais-vasco/')
    añadir_salto()
    texto("""'Pero este estudio no termina aquí: con esta información la ciudad quiere establecer un modelo predictivo a través del cual, puedan predecir para determinados momentos del día y la semana, si el tráfico en el polígono será elevado, y establecer políticas que mejoren los accesos y el tránsito en estas zonas. Y es aquí donde comienza tu labor: mediante los datos de train que encontrarás en el apartado de “Datos”, deberás:    
    \n1. Analizar la información facilitada en el apartado de “Datos” y estudiar la calidad de la información recopilada por el sistema de visión a la entrada del polígono.
    \n2. A través de los datos de entrenamiento (train) desarrollar un modelo de clasificación que, en función de las variables de estudio que consideres más importantes, determine si el tráfico es muy denso, alto, normal o bajo (de acuerdo a la clasificación que encontrarás en el apartado “Datos”)
    Una vez tengas el modelo entrenado, con los datos de test, podrás aplicar en ellos la lógica de tu modelo, y obtener la clasificación para esos datos no etiquetados. 
    \n3. Ese resultado será el que deberás subir a Kopuru, para que podamos evaluar tu porcentaje de acierto.
    ¿Quieres deslumbrar? Si además crees que de tu análisis puede obtenerse más información, adjunta un PDF donde nos cuentes: por qué tu solución es la mejor, y que ideas se te ocurren para aplicar soluciones que mejoren el tráfico, basado en las conclusiones que has obtenido al entrenar tu modelo. Nos encantará ver soluciones que además, aporten un valor a problemas reales.'""", font_size=15, formato='i')
    st.divider()
    texto("Cargar los datos", formato='b')
    añadir_salto()
    # Cargar el X_test con el uploader
    X_test_bytes_traffic = st.file_uploader("Sube el archivo **X_test** en formato csv", type=['csv'])

    if X_test_bytes_traffic is not None:
        # Instanciamos el dataset pasandolo por el método read_csv
        X_test_raw = load_df(X_test_bytes_traffic)
        # Verificamos que haya algo dentro
        verificar_dataset_vacio(X_test_raw)
        # Verificar que no haya columna label o target o class
        verificar_no_class(X_test_raw, ['traffic situation'])
        # Verificar que el nombre de las columnas sea el esperado
        verificar_columnas_correctas(X_test_raw, COLUMNAS_CORRECTAS)
        # Verificamos que los valores sean los correcots
        #verificar_valores_concretos(X_test_raw, VALORES_CORRECTOS)
        # Preprocesamos el dataset para poder pasarlo por el modelo y lo guardamos en X_test
        try:
            X_test = preprocess_traffic(X_test_raw)
        except Exception as e:
            st.error(f"Se ha producido un error procesando **X_test**. Revisa el dataset. Si el error persiste contacta con tejedor.moreno@gmail.com. Error: {e}")
            st.stop()
        st.success('OK')
        # Guardamos en la sesión. Guardamos también el nombre del archivo original
        if st.session_state.get("traffic") is None:
            st.session_state["traffic"] = {}
        st.session_state["traffic"].update({
            "X_test_raw": X_test_raw,
            "X_test": X_test,
            "X_test_filename": os.path.splitext(X_test_bytes_traffic.name)[0]
        })
        # Posibilidad de mostrar el dataframe
        if st.toggle("Visualizar **X_test**"):
            st.dataframe(X_test_raw, use_container_width=True)        

    # Modelo
    if (X_test:=st.session_state.get("traffic", {}).get("X_test", None)) is not None:
        st.divider()
        texto("Visualizar", formato='b')
        with st.expander(f"Expande para visualizar un mapa de calor **X_test**"):
            X_test_raw = st.session_state.get("traffic", {}).get("X_test_raw")
            # Ploteamos el map de calor
            plot = plot_densidad_trafico(X_test_raw)
            st.pyplot(plot)

        st.divider()
        texto("Predecir", formato='b')
        # TODO: Meter desplegable para elegir entre más modelos ?
        model = load_model('models/traffic_xgboost_STM.pkl')
        
        añadir_salto()
        # Mostrar detalles del modelo
        with st.expander("Ver detalles del modelo **XGBoost**"):
            plt = plot_arbol_decision(model)
            texto("Primera rama", formato='b', font_size=20, centrar=True)
            st.pyplot(plt)
            plt = plot_arbol_decision(model, 1)
            texto("Segunda rama", formato='b', font_size=20, centrar=True)
            st.pyplot(plt)
            texto("(...)", formato='b', font_size=20, centrar=True)

        inferir_btn = st.button("Predecir")
        if inferir_btn:            
            # Guardamos en sesión y_preds y y_preds_raw
            with st.spinner("Calculando..."):
                try:
                    y_preds, y_prob = inferir_multiclass(model, X_test, "Traffic Situation")
                    y_preds_raw = y_preds.replace(inverted_labels_map)
                    st.session_state["traffic"].update({"y_preds": y_preds,
                                                    "y_preds_raw": y_preds_raw,
                                                    "y_prob": y_prob})
                except Exception as e:
                    st.error(f"Se ha producido un error al lanzar las predicciones: {e}")
                    st.stop()
            st.success("Inferencia completada correctamente.")

        if (y_preds:=st.session_state.get("traffic", {}).get("y_preds")) is not None:
            y_preds_raw = st.session_state.get("traffic", {}).get("y_preds_raw")
            X_test_raw = st.session_state.get("traffic", {}).get("X_test_raw")
            X_test_filename:str = st.session_state.get("traffic", {}).get("X_test_filename")
            # Posibilidad de visualizar y_preds
            if st.toggle("Visualizar **y_preds**"):
                st.dataframe(y_preds_raw, width=200, hide_index=False)            
            # Boton para descargar                
            st.download_button(
                label="Descargar predicciones",
                data=y_preds_to_csv(X_test_raw, y_preds_raw, "Traffic Situation"),
                file_name= FILENAME_CSV,
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
                verificar_valores_concretos(y_test_raw, list(labels_map))
                # Verificar que haya el mismo numero de valores que en y_preds
                verificar_cantidad_registros(y_test_raw, y_preds_raw)
                st.success('OK')
                # Label Encoder. Mapeamos 
                y_test = codificar_labels(y_test_raw, labels_map)
                # Guardamos en la sesión
                st.session_state["traffic"].update({
                    "y_test": y_test,
                    "y_test_raw": y_test_raw})

            # Evaluación y_preds vs y_test
            if (y_test:=st.session_state.get("traffic", {}).get("y_test")) is not None:
                y_prob = st.session_state.get("traffic", {}).get("y_prob")
                y_test_raw = st.session_state.get("traffic", {}).get("y_test_raw")
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
                    plt = plot_roc_auc_multiclass(y_test, y_prob, inverted_labels_map)
                    st.pyplot(plt)
                añadir_salto(2)
                col1, col2, col3 = st.columns(3)
                with col2:
                    texto("Otras métricas", formato='b', font_size=20)
                    precision, recall, f1, mcc = computar_otras_metricas(y_test, y_preds, list(inverted_labels_map))
                    col1, col2,  = st.columns(2)
                    with col1:
                        st.metric("Precision", f"{precision:.2%}")
                        st.metric("Recall", f"{recall:.2%}")
                    with col2:
                        st.metric("F1 score", f"{f1:.2%}")
                        st.metric("MCC", f"{mcc:.2%}")

if __name__ == '__main__':
    traffic_model()