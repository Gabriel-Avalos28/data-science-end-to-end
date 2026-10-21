import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# El archivo tiene 59 millones de filas (13 GB descomprimido), no entra en memoria.
# Lo leo por partes (chunks): de cada parte cuento los totales exactos y
# guardo una muestra al azar del 1% para los graficos.
RUTA = "datos/obt_sales.csv.gz" #pandas lee el .gz directo, no hay que descomprimirlo
COLUMNAS = [
    "DATE", "ITEM_ID", "CAT_ID", "STORE_ID", "STATE_ID", "UNITS", "SELL_PRICE",
    "REVENUE", "WEEKDAY", "IS_WEEKEND", "MONTH", "YEAR", "IS_EVENT_DAY", "IS_SNAP",
    "IS_PRE_LAUNCH", "IS_STORE_CLOSED", "IS_SALES_SPIKE", "IS_SUSPECTED_STOCKOUT_CONSERVATIVE",
]

# TODO: reemplaza por el nombre de la columna que tu modelo va a predecir.
OBJETIVO = "UNITS" #unidades vendidas de un producto en una tienda en un dia

total_filas = 0
faltantes = 0
ceros = 0
pre_lanzamiento = 0
tienda_cerrada = 0
sin_stock = 0
muestras = []

for parte in pd.read_csv(RUTA, usecols=COLUMNAS, chunksize=2_000_000):
    total_filas += len(parte)
    faltantes = faltantes + parte.isna().sum() #va sumando los vacios de cada columna
    ceros += (parte[OBJETIVO] == 0).sum()
    pre_lanzamiento += parte["IS_PRE_LAUNCH"].sum()
    tienda_cerrada += parte["IS_STORE_CLOSED"].sum()
    sin_stock += parte["IS_SUSPECTED_STOCKOUT_CONSERVATIVE"].sum()
    muestras.append(parte.sample(frac=0.01, random_state=42)) #1% al azar de cada parte

datos = pd.concat(muestras, ignore_index=True)

print("----- TOTALES EXACTOS DEL ARCHIVO COMPLETO -----")
print("Filas totales:", total_filas)
print("Faltantes por columna:")
print(faltantes)
print(f"Dias con {OBJETIVO} = 0: {ceros} ({ceros / total_filas:.1%})")
print(f"Filas antes del lanzamiento del producto: {pre_lanzamiento} ({pre_lanzamiento / total_filas:.1%})")
print(f"Filas con tienda cerrada: {tienda_cerrada}")
print(f"Filas con sospecha de falta de stock: {sin_stock} ({sin_stock / total_filas:.1%})")

print("----- DESDE AQUI TODO ES CON LA MUESTRA DEL 1% -----")

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))

# Paso 2: distribucion de la variable objetivo.
# Con un histograma normal todo cae en la primera barra: casi todo esta entre 0 y 5
# pero el maximo es 544. Por eso hago dos graficos y sin los dias antes del lanzamiento.
vendiendo = datos[~datos["IS_PRE_LAUNCH"]] #el ~ es para negar, me quedo con los que no son pre lanzamiento
conteo = vendiendo[OBJETIVO].value_counts().sort_index().loc[0:20] #cuantos dias con 0, 1, 2... hasta 20 unidades

fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))
izq.bar(conteo.index, conteo.values)
izq.set_title(f"{OBJETIVO} de 0 a 20")
izq.set_xlabel("Unidades vendidas en el dia")
izq.set_ylabel("Numero de dias")

der.hist(vendiendo[OBJETIVO], bins=60)
der.set_yscale("log") #escala logaritmica para que se vea la cola de valores grandes
der.set_title(f"{OBJETIVO} completo (eje y en escala log)")
der.set_xlabel("Unidades vendidas en el dia")
der.set_ylabel("Numero de dias (log)")

plt.tight_layout()
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[OBJETIVO] < q1 - 1.5 * iqr) | (datos[OBJETIVO] > q3 + 1.5 * iqr)]
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10))

# TODO, paso 4: elige la variable de entrada que esperas que mas influya
# en el objetivo y grafica su relacion con el (dispersion si es numerica,
# promedio del objetivo por grupo si es categorica).
# Creo que el dia de la semana influye mucho: la gente compra mas el fin de semana.
orden_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
por_dia = datos.groupby("WEEKDAY")[OBJETIVO].mean().reindex(orden_dias)
print(por_dia.round(2))

plt.figure()
plt.bar(por_dia.index, por_dia.values)
plt.title("Unidades promedio por dia de la semana")
plt.ylabel("Unidades promedio")
plt.xticks(rotation=45)
plt.tight_layout() #para que no se corten los nombres de los dias
plt.savefig("figuras/unidades_por_dia.png")
plt.close()

# TODO, paso 5: agrega aqui las consultas que necesites para confirmar
# cada uno de tus 3 hallazgos (un filtro, un groupby, una correlacion).

# Hallazgo 1: los ceros antes del lanzamiento no son ventas reales
print(datos.groupby("IS_PRE_LAUNCH").agg(
    filas=(OBJETIVO, "count"),
    unidades_promedio=(OBJETIVO, "mean"),
    precio_vacio=("SELL_PRICE", lambda p: p.isna().sum()),
))

# Hallazgo 2: el objetivo esta muy concentrado en pocos valores
print(datos[OBJETIVO].value_counts().head(6)) #cuantas filas tienen 0, 1, 2...
print("Promedio:", round(datos[OBJETIVO].mean(), 2), "Mediana:", datos[OBJETIVO].median())
print(datos.groupby("CAT_ID")[OBJETIVO].mean().round(2)) #por categoria

# Hallazgo 3: REVENUE es UNITS por SELL_PRICE, no se puede usar para predecir
print(datos[[OBJETIVO, "REVENUE", "SELL_PRICE"]].corr()[OBJETIVO].round(3))
print(datos.groupby("IS_SNAP")[OBJETIVO].mean().round(2)) #dias con cupones de comida
