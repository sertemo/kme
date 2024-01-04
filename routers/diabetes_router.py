"""Script que recoge la lógica del desafío relacionado con el Hill vs Valley."""
# paquetes integrados
from typing import Any
# paquetes de terceros
import streamlit as st
from streamlit_utils import texto, imagen_con_enlace, añadir_salto
# mis propios paquetes



def diabetes_model():
    texto("En construcción", centrar=True, color='#ccc71d') #! A Borrar cuando se termine de desarrollar. GitFlow 
    imagen_con_enlace('https://i.imgur.com/RsW32VA.jpg', 'https://kopuru.com/challenge/diabetes-challenge-de-entrenamiento/')
    añadir_salto()
    texto("""'La diabetes es una enfermedad crónica que se caracteriza por generar niveles altos de glucosa en sangre. 
        Esta, puede producir daños en el corazón, los vasos sanguíneos, los riñones, y los nervios. 
        Existen diversos factores que pueden generarla, estudiarlos brevemente te ayudará a solucionar el problema.<br>
        En este desafío, te pedimos que construyas un modelo predictivo que responda a la pregunta: 
        “¿Tiene diabetes?” utilizando los datos que puedan resultar clave para diagnosticar la diabetes.<br>
        <br>
        ¡IMPORTANTE! En este dataset solo hay información de mujeres mayores de 21 años. 
        Como se trata de un ejercicio de prueba/entrenamiento, te proponemos que lo resuelvas 
        utilizando el clasificador Naive-Bayes y 10 fold cross validation.'""", font_size=15, formato='i')
    st.divider()
    texto("Cargar los datos", formato='b')
    añadir_salto()