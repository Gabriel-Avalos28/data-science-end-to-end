import pandas as pd
import os
import matplotlib.pyplot as plt

os.makedirs('./starter/figuras', exist_ok=True)

data = pd.read_csv('./starter/viajes_taxi.csv')

print('Analisis exploratorio descriptivo')

data.info()

print(data.dtypes)
print(data.isna().sum())  #el isna es un metodo de la clase dataframe que me permite identificar los valores 
print(data.duplicated().sum())

print(data.describe())  #Podemos correr todo esto a través de docller

# --------- HACER GRAFICO CON MATPLOTLIB

plt.figure() #Es el concepto del grafico como un activo, todo lo que defina abajo sera una sola figura
plt.hist(data['tarifa'], bins=50) #Histograma de cada columna de la data,BINS ES PARA DIVIDIR EN 10 PARTES LA VARIABLE 
plt.title('Distribución de tarifas')
plt.xlabel('USD') # TARIFA EN USD, del eje y
plt.ylabel('Numero de viajes') # Numero de viajes
plt.savefig('./starter/figuras/hist_tarifa.png') #Guarda la figura en starter
plt.close()

plt.figure() #Es el concepto del grafico como un activo, todo lo que defina abajo sera una sola figura
plt.boxplot(data['tarifa']) #Histograma de cada columna de la data,BINS ES PARA DIVIDIR EN 10 PARTES LA VARIABLE 
plt.title('Boxplot de la tarifa')
plt.savefig('./starter/figuras/boxplot_tarifa.png') #Guarda la figura en starter
plt.close()

# -------------- EXTRACCION DE VALORES ATIPICOS

q1 = data['tarifa'].quantile(0.25)
q3 = data['tarifa'].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

atipicos = data.query(f' tarifa < {limite_inferior} or tarifa > {limite_superior}') #query me permite hacer consultas de forma natural como si fuera un WHERE en SQL
sin_atipicos = data.query(f' tarifa >= {limite_inferior} and tarifa <= {limite_superior}')

print('Valores atipicos')
print(atipicos) #LOS VALORES ATIPICOS SON PARA ANALIZARLOS PORQUE TE ESTAN DICIENDO QUE PASA ALGO
print('Cantidad de atipicos:', len(atipicos))

# -------------- REVISAR SI LOS ATIPICOS SON ERRORES O NO

print('Tarifas de 999.99:', (data['tarifa'] == 999.99).sum()) #esto es claramente un error de captura
print('Duraciones negativas:', (data['duracion_min'] < 0).sum()) #no existe un viaje que dure menos de 0 minutos

print(atipicos['zona_origen'].value_counts()) #vemos de donde salen los viajes atipicos
print(atipicos['zona_destino'].value_counts()) #y a donde llegan, muchos son del aeropuerto JFK

#------------AHORA PODEMOS VER LAS CORRELACIONES (PEARSON-> ASUME LINEALIDAD, SPEARMAN-> CUANDO NO ASUME LINEALIDAD)

plt.figure()
plt.scatter(sin_atipicos['distancia_km'], sin_atipicos['tarifa']) #distancia en el eje x, tarifa en el eje y
plt.title('Dispercion de distancia_km vs tarifa')
plt.xlabel('Distancia (km)')
plt.ylabel('Tarifa (USD)')
plt.savefig('./starter/figuras/dispercion_distancia_vs_tarifa_sinatipicos.png')
plt.close()

print(data.corr(numeric_only=True)) #solo con las columnas numericas, si no da error por las columnas de texto

