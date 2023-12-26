"""Script con funciones auxiliares para el entorno de Streamlit
"""
import streamlit as st

DEFAULT_COLOR = '#000000'

def texto(texto:str, /, *, font_size:int=30, color:str=DEFAULT_COLOR, font_family:str="Helvetica", formato:str="") -> None:
    """ Función para personalizar el texto con HTML"""
    if formato:
        texto = f"<{formato}>{texto}</{formato}>"
    texto_formateado = f"""<div style='font-size: {font_size}px; color: {color}; font-family: {font_family}'>{texto}</div>"""
    st.markdown(texto_formateado, unsafe_allow_html=True)

def mostrar_enlace(label:str, url:str, *, color:str=DEFAULT_COLOR, font_size:str='16px') -> None:
    """Muestra un enlace personalizado.

    Args:
    label (str): El texto que se mostrará como el enlace.
    url (str): La URL a la que apunta el enlace.
    color (str): Color del texto del enlace.
    font_size (str): Tamaño del texto del enlace.
    """
    html = f'<a href="{url}" target="_blank" style="color: {color}; font-size: {font_size}; text-decoration: none;">{label}</a>'
    st.markdown(html, unsafe_allow_html=True)

def añadir_salto(num_saltos:int=1) -> None:
    """Añade <br> en forma de HTML para agregar espacio
    """
    saltos = f"{num_saltos * '<br>'}"
    st.markdown(saltos, unsafe_allow_html=True)

def imagen_con_enlace(url_imagen, url_enlace, alt_text="Imagen", max_width:int=100):
    """Muestra una imagen que es también un hipervínculo en Streamlit.

    Args:
    url_imagen (str): URL de la imagen a mostrar.
    url_enlace (str): URL a la que el enlace de la imagen debe dirigir.
    alt_text (str): Texto alternativo para la imagen.
    """
    html = f'<a href="{url_enlace}" target="_blank"><img src="{url_imagen}" alt="{alt_text}" style="max-width:{max_width}%; height:auto;"></a>'
    st.markdown(html, unsafe_allow_html=True)