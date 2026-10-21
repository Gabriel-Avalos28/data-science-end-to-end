import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

# TODO 1: quita solo los errores que el Ejercicio 1 ya confirmo (tarifa
# 999.99 y duracion negativa). No quites los demas atipicos. Imprime
# cuantas filas quedan.
limpios = viajes.query('tarifa != 999.99 and duracion_min >= 0') #solo quito los errores, los viajes de JFK se quedan
print('Filas que quedan:', len(limpios))

# TODO 2: grafico de dispersion de distancia_km (eje x) contra tarifa
# (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
# antes de seguir: busca puntos que no sigan la tendencia general y
# cuentalos con un filtro.
plt.figure()
plt.scatter(limpios['distancia_km'], limpios['tarifa'], s=5) #s es el tamaño de los puntos
plt.title('Distancia vs tarifa')
plt.xlabel('Distancia (km)')
plt.ylabel('Tarifa (USD)')
plt.savefig('figuras/distancia_vs_tarifa.png')
plt.close()

# en el grafico se ven dos grupos que no siguen la tendencia:
# una linea horizontal en 70 (tarifa fija de JFK) y una columna en distancia 0
tarifa_fija = limpios.query('tarifa == 70')
print('Viajes con tarifa de 70:', len(tarifa_fija))
print(tarifa_fija['zona_origen'].value_counts().head(3)) #ver si son de JFK
print(tarifa_fija['zona_destino'].value_counts().head(3))

distancia_cero = limpios.query('distancia_km == 0')
print('Viajes con distancia 0:', len(distancia_cero))
print(distancia_cero[['distancia_km', 'duracion_min', 'tarifa']].describe()) #si tienen duracion y tarifa, el viaje si existio

# TODO 3: propina promedio y cantidad de viajes con propina 0, por
# metodo de pago (groupby + agg).
propinas = limpios.groupby('metodo_pago', dropna=False).agg( #dropna=False para ver tambien los que no tienen metodo de pago
    propina_promedio=('propina', 'mean'),
    viajes_sin_propina=('propina', lambda p: (p == 0).sum()),
    total_viajes=('propina', 'count'),
)
print(propinas)

# TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina
# con la tarifa.
print(limpios[['distancia_km', 'duracion_min', 'pasajeros', 'propina', 'tarifa']].corr()['tarifa']) #solo me quedo con la columna de tarifa

# TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.
print('Pasajeros vacio:', limpios['pasajeros'].isna().sum())
print('Pasajeros en 0:', (limpios['pasajeros'] == 0).sum())
