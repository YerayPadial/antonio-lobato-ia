
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# configuro la pagina para que se vea mas profesional
st.set_page_config(page_title="Tasación Ford IA", page_icon="🚗", layout="centered")

# cargo el modelo entrenado
# uso el decorador cache para no recargar el modelo cada vez que se interactua
@st.cache_resource
def load_model():
    return joblib.load('modelo_ford_pipeline.pkl')

model = load_model()

# titulo y descripcion de la app
st.title("Predicción de Precio de Coches Ford")
st.write("""
Introduce las características del vehículo para obtener una estimación de su precio de mercado.
Esta aplicación utiliza un modelo de Machine Learning (Random Forest) entrenado con datos históricos.
""")

# creo el formulario en la barra lateral para que quede mas limpio
st.sidebar.header("Características del Vehículo")

# he puesto los modelos mas comunes del dataset ford
model_options = ['Fiesta', 'Focus', 'Puma', 'Kuga', 'EcoSport', 'Mondeo', 'Ka+', 'C-MAX', 'S-MAX', 'Galaxy', 'Edge', 'Mustang']
transmission_options = ['Manual', 'Automatic', 'Semi-Auto']
fuel_options = ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other']

# inputs del usuario
selected_model = st.sidebar.selectbox("Modelo", model_options)
year = st.sidebar.slider("Año de matriculación", 2005, 2025, 2019)
transmission = st.sidebar.selectbox("Transmisión", transmission_options)
mileage = st.sidebar.number_input("Kilómetros (Millaje)", min_value=0, value=20000, step=1000)
fuel_type = st.sidebar.selectbox("Combustible", fuel_options)

# inputs tecnicos (con valores por defecto promedios para no bloquear al usuario)
st.sidebar.subheader("Datos Técnicos")
tax = st.sidebar.number_input("Impuesto (£)", min_value=0, value=150)
mpg = st.sidebar.number_input("Consumo (MPG)", min_value=0.0, value=55.0)
engine_size = st.sidebar.number_input("Motor (Litros)", min_value=0.0, value=1.0, step=0.1)

# boton para calcular
if st.sidebar.button("Calcular Precio"):
    # creo un dataframe con los datos de entrada
    # es importante que las columnas se llamen IGUAL que en el entrenamiento
    input_data = pd.DataFrame({
        'model': [selected_model],
        'year': [year],
        'price': [0], 
        'transmission': [transmission],
        'mileage': [mileage],
        'fuelType': [fuel_type],
        'tax': [tax],
        'mpg': [mpg],
        'engineSize': [engine_size]
    })

    # aseguro que solo paso las columnas que el modelo espera (quitando price si estaba)
    X_input = input_data.drop('price', axis=1, errors='ignore')

    # prediccion
    try:
        prediction = model.predict(X_input)[0]
        st.balloons() # un toque visual para el 10
        st.success(f"### Precio Estimado: £{prediction:,.2f}")
        st.caption("Este precio es una estimación basada en 17,000 coches analizados.")
    except Exception as e:
        st.error(f"Error en el cálculo: {e}")
