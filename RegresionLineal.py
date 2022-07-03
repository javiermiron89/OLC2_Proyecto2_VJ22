import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def RegLin(data):
    #SE SOLICITAN LOS DATOS
    col1, col2 = st.columns(2)
    with col1:
        ejeX = st.selectbox(
            'Seleccione el Eje X',
            (data.columns.values)
        )

    with col2:
        ejeY = st.selectbox(
            'Seleccione el Eje Y',
            (data.columns.values)
        )
    
    correlativo = st.number_input('Inserte la prediccion', step = 1)

    if st.button('Calcular Resultados'):
        st.write("## Paso 4: Resultados")
        #SE REALIZAN LOS CALCULOS PARA LA REGRESION LINEAL
        x = np.asarray(data[ejeX]).reshape(-1,1)
        y = data[ejeY]

        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        y_pred = regr.predict(x)

        # Resultados
        coeficiente = regr.coef_
        intercepto = regr.intercept_
        prediccion = regr.predict([[int(correlativo)]])
        errorCuadratico = mean_squared_error(y, y_pred)
        coeficienteDeterminacion = r2_score(y, y_pred)

        
        with st.expander("Graficar puntos"):
            figura, ax = plt.subplots()
            ax.scatter(x, y, color='black')
            plt.title('Regresio lineal\nCoeficiente de regresion: ' + str(coeficiente))
            plt.xlabel(ejeX)
            plt.ylabel(ejeY)
            plt.grid()
            st.pyplot(figura)

        with st.expander("Grafica de tendencia"):
            figura, ax = plt.subplots()
            ax.scatter(x, y, color='black')
            ax.plot(x, y_pred, color='blue', linewidth=3)
            plt.title('Regresio lineal\nCoeficiente de regresion: ' + str(coeficiente))
            plt.xlabel(ejeX)
            plt.ylabel(ejeY)
            plt.grid()
            st.pyplot(figura)
        
        with st.expander("Funcion de tendencia"):
            #y = mx + b
            strCoeficiente = str(coeficiente)
            strCoeficiente = strCoeficiente.replace("[", "")
            strCoeficiente = strCoeficiente.replace("]", "")
            st.info("**F(x)** = " + strCoeficiente + "**x** + " + str(intercepto))

        with st.expander("Prediccion de tendencia"):
            strPrediccion = str(prediccion)
            strPrediccion = strPrediccion.replace("[", "")
            strPrediccion = strPrediccion.replace("]", "")
            st.success("**" + strPrediccion + "**")
