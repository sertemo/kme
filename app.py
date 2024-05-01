from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tensorflow floating numbers

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_utils import (texto, 
                añadir_salto, 
                mostrar_enlace,
                imagen_con_enlace 
                )
from routers.tictactoe_router import tictactoe_model
from routers.traffic_router import traffic_model
from routers.hillvalley_router import hillvalley_model
from routers.diabetes_router import diabetes_model
from routers.aidtec.models import SerializableClassifier
from routers.aidtec.aidtec_router import aidtec_model

# Configuración de la app
st.set_page_config(
    page_title=f"KME Evaluación de Modelos de Desafíos Kopuru",
    page_icon="🧠", #🎇
    layout="wide",
    initial_sidebar_state="auto",
)
VERSION = '0.5.0'
etiqueta_version = f"""
        <span style="
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        background-color: #000000;
        padding: 5px 10px;
        border-radius: 4px;
        ">
            {VERSION}
        </span>
        """

def main():
    with st.sidebar:
        # Logo de la app        
        imagen_con_enlace('https://i.imgur.com/RxmHMZa.png', 'https://kopuru.com/', centrar=True, max_width=70)
        texto(f"🧠KME", formato='b', font_size=50, font_family="Courier", centrar=True)
        texto("Kopuru Model Evaluation", font_size=15, formato='b', centrar=True, color='#a3a3a3')
        
        añadir_salto()
        seleccion_menu = option_menu(
            menu_title='Desafío', # texto("Desafío", centrar=True, formato='b', font_size=20),
            options=[
                "Tic Tac Toe", 
                "Hill Valley",
                "Traffic Prediction",
                "Diabetes",
                "AidTec Solutions"
            ],
            default_index=4,
            icons=[ #lista de iconos aqui: https://icons.getbootstrap.com/
                "grid-3x3",
                "symmetry-horizontal",
                "car-front",
                "capsule",
                "funnel"
            ],
            menu_icon="lightbulb",
            )
        
        st.divider()
        _, col2, col3, col4, _ = st.columns(5)            
        with col2:
            imagen_con_enlace('https://i.imgur.com/umyrYj9.png',
                            'https://github.com/sertemo', 
                            alt_text='GitHub')
        with col3:
            imagen_con_enlace('https://i.imgur.com/hLAeokj.png', 
                            'https://www.linkedin.com/in/stm84/', 
                            alt_text='linkedin')
        with col4:
            imagen_con_enlace('https://i.imgur.com/Qc8t46o.png', 
                            'https://stm-cv.streamlit.app/', 
                            alt_text='Chat-CV')

        texto('tejedor.moreno@gmail.com', centrar=True, font_size=10)
        texto(f"v{VERSION} {datetime.strftime(datetime.now(), format='%m/%Y')}", centrar=True, font_size=10)

    if seleccion_menu == "Tic Tac Toe":
        tictactoe_model()
    elif seleccion_menu == "Traffic Prediction":
        traffic_model()
    elif seleccion_menu == "Hill Valley":
        hillvalley_model()
    elif seleccion_menu == "Diabetes":
        diabetes_model()
    elif seleccion_menu == "AidTec Solutions":
        aidtec_model()


if __name__ == '__main__':
    main()
    st.session_state