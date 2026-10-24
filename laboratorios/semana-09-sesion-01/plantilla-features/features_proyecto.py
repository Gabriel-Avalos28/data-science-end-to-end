import pandas as pd

# Como corre en Docker (desde la carpeta semana-09-sesion-01):
#   docker compose run --rm -w /app/plantilla-features python-app python features_proyecto.py

# TODO: la misma ruta que usaste en la exploracion de la Semana 8.
# El docker-compose conecta la carpeta datos de la Semana 8 en /datos.
RUTA = "/datos/obt_sales.csv.gz"

# TODO: la columna que tu modelo va a predecir.
OBJETIVO = "UNITS"

# Igual que en la Semana 8: 59 millones de filas no entran en memoria, asi que leo por partes.
# Separo entrenamiento y prueba por fecha ANTES de calcular nada: entreno con todo hasta 2015
# y pruebo con 2016. El precio promedio de cada producto lo calculo solo con entrenamiento,
# si no, se cuela informacion de prueba (data leakage).
CORTE = "2016-01-01"
SERIE = ["ITEM_ID", "STORE_ID"] #una serie es un producto en una tienda
COLUMNAS = ["DATE", "ITEM_ID", "STORE_ID", "CAT_ID", "UNITS", "SELL_PRICE", "IS_PRE_LAUNCH"]

lanzamientos = []
precios = []
muestras = []
fecha_min, fecha_max = "9999", "0000"

for parte in pd.read_csv(RUTA, usecols=COLUMNAS, chunksize=2_000_000):
    fecha_min = min(fecha_min, parte["DATE"].min())
    fecha_max = max(fecha_max, parte["DATE"].max())
    vendiendo = parte[~parte["IS_PRE_LAUNCH"]] #solo los dias en que el producto ya estaba a la venta

    # primer dia a la venta de cada producto en cada tienda
    lanzamientos.append(vendiendo.groupby(SERIE)["DATE"].min())

    # suma y cantidad de precios, solo con entrenamiento, para sacar el promedio al final
    entrenamiento = vendiendo[vendiendo["DATE"] < CORTE]
    precios.append(entrenamiento.groupby(SERIE)["SELL_PRICE"].agg(["sum", "count"]))

    muestras.append(vendiendo.sample(frac=0.01, random_state=42)) #1% al azar, igual que en la Semana 8

# junto lo de todas las partes
lanzamiento = pd.concat(lanzamientos).groupby(level=[0, 1]).min().rename("fecha_lanzamiento")
suma_precios = pd.concat(precios).groupby(level=[0, 1]).sum()
precio_promedio = (suma_precios["sum"] / suma_precios["count"]).rename("precio_promedio")

datos = pd.concat(muestras, ignore_index=True)
datos = datos.join(lanzamiento, on=SERIE).join(precio_promedio, on=SERIE)

print("Fechas del archivo:", fecha_min, "a", fecha_max)
print("Filas de la muestra (sin dias antes del lanzamiento):", len(datos))
print("Productos-tienda:", len(lanzamiento))

# TODO, feature 1: crea la columna a partir de una o mas columnas
# existentes.
# dias_desde_lanzamiento: cuantos dias lleva el producto a la venta en esa tienda.
# Sale del Hallazgo 1 de la Semana 8 (IS_PRE_LAUNCH): un producto nuevo vende distinto.
datos["dias_desde_lanzamiento"] = (pd.to_datetime(datos["DATE"]) - pd.to_datetime(datos["fecha_lanzamiento"])).dt.days

# TODO, feature 2.
# precio_relativo: el precio del dia dividido para el precio promedio del producto.
# Es un ratio, como dice el articulo: 0.9 significa que esta 10% mas barato de lo normal.
datos["precio_relativo"] = datos["SELL_PRICE"] / datos["precio_promedio"]

# Evidencia: compara el objetivo segun cada feature nueva.
# Toda la evidencia la saco solo con entrenamiento, para no mirar los datos de prueba.
entreno = datos[datos["DATE"] < CORTE]
print("Filas de entrenamiento en la muestra:", len(entreno), "| de prueba:", len(datos) - len(entreno))

# Feature 1: numerica, su correlacion con el objetivo
print(entreno[["dias_desde_lanzamiento", OBJETIVO]].corr().round(3))
# como la relacion puede no ser lineal, tambien el promedio por grupos de antiguedad
grupos_edad = pd.cut(entreno["dias_desde_lanzamiento"], bins=[-1, 30, 90, 365, 10_000],
                     labels=["0-30 dias", "31-90 dias", "91-365 dias", "mas de 1 anio"])
print(entreno.groupby(grupos_edad, observed=True)[OBJETIVO].agg(["mean", "count"]).round(2))

# Feature 2: numerica, su correlacion con el objetivo
print(entreno[["precio_relativo", OBJETIVO]].corr().round(3))
# y el promedio por grupos: en oferta, precio normal o mas caro
grupos_precio = pd.cut(entreno["precio_relativo"], bins=[0, 0.95, 1.05, 100],
                       labels=["oferta (<0.95)", "normal", "mas caro (>1.05)"])
print(entreno.groupby(grupos_precio, observed=True)[OBJETIVO].agg(["mean", "count"]).round(2))
# el promedio general esconde diferencias entre categorias, asi que tambien por categoria
print(entreno.groupby(["CAT_ID", grupos_precio], observed=True)[OBJETIVO].mean().unstack().round(2))

print(datos.head())
