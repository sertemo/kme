import streamlit as st
from streamlit_option_menu import option_menu

from streamlit_utils import (texto, 
                añadir_salto, 
                mostrar_enlace,
                imagen_con_enlace 
                )
from routers.tictactoe_router import tictactoe_model

# Configuración de la app
st.set_page_config(
    page_title=f"KME Evaluación de Modelos de Desafíos Kopuru",
    page_icon="🎇",
    layout="wide",
    initial_sidebar_state="auto",
)
VERSION = '1.0'
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
        texto(f"KME {etiqueta_version}", formato='b', font_size=60, font_family="Courier")
        texto("Kopuru Model Evaluation", font_size=20)
        añadir_salto()
        seleccion_menu = option_menu(
            menu_title="Desafíos",
            options=[
                "Tic Tac Toe", 
                "Hill Valley",
                "Traffic Prediction",
            ],
            default_index=0,
            icons=[ #lista de iconos aqui: https://icons.getbootstrap.com/
                "grid-3x3",
                "symmetry-horizontal",
                "car-front-fill",
            ],
            menu_icon="lightbulb",
            )
        imagen_con_enlace('https://i.imgur.com/q1JiUua.png','https://kopuru.com/', max_width=20)
        st.divider()
        st.caption("STM · 2024")
        # TODO Poner imagen y enlace a Linkedin y a app de chat-cv y GitHub

    if seleccion_menu == "Tic Tac Toe":
        tictactoe_model()
    
        
if __name__ == '__main__':
    main()
    st.session_state