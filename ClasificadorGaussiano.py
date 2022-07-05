import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

def ClaGau(data):

    variablesEntrada = list()

    opcion = st.radio(
        "¿Como deseas seleccionar las variables de entrada?",
        ('Todas', 'Seleccion Multiple')
    )

    if opcion == 'Todas':
        st.warning("Se omitira la variable de salida")
        for dato in data.columns.values:
            variablesEntrada.append(dato)
    else:
        variablesEntrada = st.multiselect(
            'Seleccione las variable de entrada',
            (data.columns.values)
        )
    
    variableSalida = st.selectbox(
            'Seleccione la variable de salida',
            (data.columns.values)
    )

    agree = st.checkbox('¿Desea utilizar LabelEncoder?')
    
    st.write("## Ingreso de valores")

    if opcion == 'Todas':
        variablesEntrada.remove(variableSalida)
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("# Valores a ingresar:")
        st.write("# " + str(len(variablesEntrada)))

    with col2:
        listaValores = list()
        valoresRestantes = len(variablesEntrada)
        cadena = st.text_input('Inserte los valores (separados por coma)', '')

        if cadena != '':
            auxiliar = cadena.split(",")
            st.info("**Valores restantes: " + str(valoresRestantes-len(auxiliar)) + "**")
            mapeo = list(map(int,auxiliar))
            mapeo = np.array(mapeo)
            listaValores = np.asarray(mapeo)
            
        """
        for i in range(valoresRestantes):
            valor = st.number_input('Inserte el valor ' + str(i) + ':', step = 1, key=i)
            listaValores.append(valor)
            valoresRestantes -= 1

        #st.write(listaValores)
        """


    if st.button('Calcular Resultados'):
        st.write("# Paso 4: Resultados")

        model = GaussianNB()

        #Se prepara un arreglo con los dataFrame de las variables seleccionadas
        dataFrameEntrada = list()
        for variable in variablesEntrada:
            temporal = data[variable]
            temporal = np.asarray(temporal)
            dataFrameEntrada.append(temporal)
        dataFrameEntrada = np.asarray(dataFrameEntrada)

        #Se preapra el arreglo de salida
        dataFrameSalida = data[variableSalida]
        
        if agree:
            # Se crea el Codificador
            le = preprocessing.LabelEncoder()
            # Se codifican los valores
            encoded = list()
            for dato in dataFrameEntrada:
                encoded.append(le.fit_transform(dato))
            label = le.fit_transform(dataFrameSalida)

            # Combinando los atributos en una lista simple de tuplas (No se toma en concideracion la tupla de N o P)
            features = list()
            longitudInterna = len(encoded[0])
            for i in range(longitudInterna):
                listTemporal = list()
                for enc in encoded:
                    listTemporal.append(enc[i])
                features.append(listTemporal)
            
            features = np.asarray(features)

            #st.write(features)

            model.fit(features, label)
        else:
            features = list()
            longitudInterna = len(dataFrameEntrada[0])
            for i in range(longitudInterna):
                listTemporal = list()
                for dfe in dataFrameEntrada:
                    listTemporal.append(dfe[i])
                features.append(listTemporal)
            features = np.asarray(features)

            model.fit(features, dataFrameSalida)

        # Crear el clasificador Gaussiano
        predicted = model.predict([listaValores])

        with st.expander("Prediccion de la Tendencia"):
            strPredicted = str(predicted)
            strPredicted = strPredicted.replace("[", "")
            strPredicted = strPredicted.replace("]", "")
            st.success('Valor de prediccion: ' + strPredicted)
