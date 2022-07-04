import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
from sklearn import tree

def ArbDec(data):
    
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

    if st.button('Calcular Resultados'):
        st.write("# Paso 4: Resultados")

        if opcion == 'Todas':
            variablesEntrada.remove(variableSalida)
        
        #Se prepara un arreglo con los dataFrame de las variables seleccionadas
        dataFrameEntrada = list()
        for variable in variablesEntrada:
            dataFrameEntrada.append(data[variable])
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
            for i in range(len(encoded)):
                listTemporal = list()
                for enc in encoded:
                    listTemporal.append(enc[i])
                    #st.write('Dato: ', enc[i])
                features.append(listTemporal)
            
            # Se encaja con el modelo (Aca se pasa como segundo parametro la tupla de N o P)
            clf = DecisionTreeClassifier().fit(features, label)

        else:
            features = list()
            for i in range(len(dataFrameEntrada)):
                listTemporal = list()
                for dfe in dataFrameEntrada:
                    listTemporal.append(int(dfe[i]))
                features.append(listTemporal)
            
            # Se encaja con el modelo (Aca se pasa como segundo parametro la tupla de N o P)
            clf = DecisionTreeClassifier().fit(features, dataFrameSalida)
            
            with st.expander("Prediccion de la Tendencia"):
                fig2, ax2 = plt.subplots()
                plot_tree(clf, filled=True)
                plt.figure(figsize=(100,100))
                st.pyplot(fig2)