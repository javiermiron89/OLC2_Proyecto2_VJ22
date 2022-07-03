import streamlit as st
from matplotlib.patches import bbox_artist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def RegPol(data):
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
    
    grado = st.number_input('Inserte el grado', value=2, step = 1)

    valorPredecir = st.number_input('Inserte la prediccion', step = 1)

    if st.button('Calcular Resultados'):
        st.write("# Paso 4: Resultados")
        #---------------------------------------------------------------------------------------------------------------
        # Paso 1: Entrenar la informacion
        X = data[ejeX]
        Y = data[ejeY]

        X = np.asarray(X)
        Y = np.asarray(Y)

        X = X[:,np.newaxis]
        Y = Y[:,np.newaxis]

        with st.expander("Grafica de Puntos"):
            figura, ax = plt.subplots()
            ax.scatter(X, Y, color='black')
            plt.title('Regresio lineal\nCoeficiente de regresion: ')
            plt.xlabel(ejeX)
            plt.ylabel(ejeY)
            plt.grid()
            st.pyplot(figura)
        
        #---------------------------------------------------------------------------------------------------------------
        # Paso 2: Preparacion de la informacion
        
        nb_degree = grado
        polynomial_feature = PolynomialFeatures(degree = nb_degree)
        X_TRANSF =  polynomial_feature.fit_transform(X)

        #---------------------------------------------------------------------------------------------------------------
        # Paso 3: Definir y entrenar el modelo

        model =  LinearRegression()
        model.fit(X_TRANSF, Y)

        #---------------------------------------------------------------------------------------------------------------
        # Paso 4: Calcular Bayes y varianza

        Y_NEW = model.predict(X_TRANSF)

        rmse = np.sqrt(mean_squared_error(Y, Y_NEW))
        r2 =  r2_score(Y, Y_NEW)

        #---------------------------------------------------------------------------------------------------------------
        # Paso 5: Prediccion

        x_new_min = 0.0
        x_new_max = valorPredecir

        X_NEW = np.linspace(x_new_min, x_new_max, 50)
        X_NEW = X_NEW[:,np.newaxis]

        X_NEW_TRANSF = polynomial_feature.fit_transform(X_NEW)
        Y_NEW = model.predict(X_NEW_TRANSF)

        # Resultados
        prediccion = Y_NEW[Y_NEW.size-1]

        with st.expander("Grafica Polinomial"):
            figura, ax = plt.subplots()
            ax.plot(X_NEW, Y_NEW, color='coral', linewidth=3)
            title1 = 'Degree = {}; RMSE= {}; R2 = {};'.format(nb_degree, rmse, r2)
            title2 = 'Prediccion = {}'.format(Y_NEW[Y_NEW.size-1])
            plt.title("Regresio Polinomial\n" + title1 + "\n" + title2)
            plt.xlim(x_new_min, x_new_max)
            plt.xlabel(ejeX)
            plt.ylabel(ejeY)
            plt.grid()
            st.pyplot(figura)

        with st.expander("Funcion de Tendencia"):
            #AnXn^n + An-1Xn-1^n-1 + ... + AX^0 + B
            valorN = grado
            coeficientes = model.coef_
            intercepto = model.intercept_

            cadena = "**F(x)** = "
            for coeficiente in coeficientes:
                for coef in reversed(coeficiente):
                    cadena += str(coef) + "**x^" + str(valorN) + "** + "
                    valorN -= 1

            strIntercepto = str(intercepto)
            strIntercepto = strIntercepto.replace("[", "")
            strIntercepto = strIntercepto.replace("]", "")

            cadena += str(strIntercepto)
            st.info(cadena)
            # strCoeficiente = str(coeficiente)

        with st.expander("Prediccion de Tendencia"):
            strPrediccion = str(prediccion)
            strPrediccion = strPrediccion.replace("[", "")
            strPrediccion = strPrediccion.replace("]", "")
            st.success("**" + strPrediccion + "**")