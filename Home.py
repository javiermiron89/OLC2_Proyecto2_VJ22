import streamlit as st
import pandas as pd
import numpy as np
import RegresionLineal as rl
import RegresionPolinomial as rp
import ClasificadorGaussiano as cg

#Variables Globales
df = ""
nombreArchivo = ""
extensionArchivo = ""
paso1 = False
paso2 = False

st.set_page_config(
    page_title="Inicio",
    page_icon="👋",
)

with st.sidebar:
    st.write("# Proyecto 2")
    st.write("# 201602694 👋")

st.write("# Paso 1: Cargar un archivo de datos")

df = st.file_uploader("Seleccione un archivo", type=('csv', 'xls', 'xlsx', 'json'))
if df:
    nombreArchivo = df.name
    extensionArchivo = nombreArchivo.split('.')[1]
    if extensionArchivo == "json":
        df = pd.read_json(df)
        paso1 = True
    elif extensionArchivo == "csv":
        df = pd.read_csv(df)
        paso1 = True
    else:
        df = pd.read_excel(df)
        paso1 = True

if st.checkbox('Mostrar Datos'):
    st.write(df)


if paso1:
    st.write("# Paso 2: Seleccion de algoritmo")
    option = st.selectbox(
        'Seleccione el algoritmo que desea ejecutar',
        ('Regresion Lineal', 'Regresion Polinomial', 'Clasificador Gaussiano', 'Clasificador de arboles de decision', 'Redes neuronales')
    )
    #st.write(option)
    paso2 = True

if paso2:
    st.write("# Paso 3: Parametrizar " + option)
    if option == "Regresion Lineal":
        rl.RegLin(df)
    elif option == "Regresion Polinomial":
        rp.RegPol(df)
    elif option == "Clasificador Gaussiano":
        cg.ClaGau(df)

