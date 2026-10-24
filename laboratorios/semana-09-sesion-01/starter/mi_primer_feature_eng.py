import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression # INTERFAZ DE APRENDIZAJE, este tiene fit.predict
from sklearn.preprocessing import OrdinalEncoder # INTERFAZ DE TRANSFORMACION, Ordinal Enconder -> tiene un metodo transform y fit de ENTRENAR

viajes = pd.read_csv('viajes_taxi.csv')

print(viajes.head())

viajes_clean = viajes.query('tarifa < 999 and duracion_min > 0')

print(viajes_clean.head())
print(viajes_clean.info())

fecha_inicio = pd.to_datetime(viajes_clean['fecha_hora_inicio'])

viajes_clean['fecha_hora_inicio'] = fecha_inicio
viajes_clean['hora'] = fecha_inicio.dt.hour
viajes_clean['dia_semana'] = fecha_inicio.dt.dayofweek

print(viajes_clean.head())

viajes_clean['hora_sin'] = np.sin(2 * np.pi * viajes_clean['hora'] / 24)
viajes_clean['hora_cos'] = np.cos(2 * np.pi * viajes_clean['hora'] / 24)

print(viajes_clean[['fecha_hora_inicio','hora','hora_sin','hora_cos']])

viajes_codificado = pd.get_dummies(viajes_clean, columns=['zona_origen','zona_destino'])

print(viajes_codificado)

encoder = OrdinalEncoder()

viajes_codificado['metodo_pago_ordinal'] = encoder.fit_transform(viajes_codificado[['metodo_pago']]).ravel() #Un dataframe en pandas esta compuesto de series, una serie es una columna

#Hay de dividir entre entrenamiento y prueba antes de todo
#Esta cayendo data que luego puede caer en prueba.abs
#Toda la transformacion, creacion de variables TODO LO QUE SE HAGA EN ENTRENAMIENTO SE VA A PRUEBAS
#NO SE SACA NINGUNA CONCLUSION DE PRUEBA


print(viajes_codificado.head())