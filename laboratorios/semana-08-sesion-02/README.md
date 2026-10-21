# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas ya lo tienes desde la Semana 4. Instala matplotlib:

```
pip install matplotlib
```

Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. También puedes correr el mismo código en un notebook de Jupyter, donde el gráfico aparece debajo de la celda.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

Básicamente, ¿con qué datos estoy trabajando? El artículo lo divide en tres pasos:

1. Resumen general: cuántas filas y columnas hay, de qué tipo son, cuántos faltantes y cuántos duplicados.
2. Cómo es cada variable y cómo se relaciona con las demás: su distribución, valores fuera de rango, y qué variables se parecen o influyen en el objetivo.
3. Calidad de los datos: faltantes, datos desbalanceados, columnas llenas de ceros, duplicados y variables repetidas o que filtran información (data leakers).

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

- Una sola variable (análisis univariado): estadísticas descriptivas y gráficos según el tipo. Para numéricas, histograma (con media, desviación y cuantiles); para categóricas, gráfico de barras con la tabla de frecuencias.
- Relación entre dos (análisis multivariado): gráficos de interacción, que básicamente son de dispersión entre dos variables, y correlaciones, que se ven en una matriz o en un mapa de calor (heatmap).

Para hacer todo esto de una vez recomienda la librería `ydata-profiling`, que genera un reporte completo con una sola línea de código.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

El modelo aprende mal, porque esos 15 valores tan grandes jalan el promedio hacia arriba y hacen que prediga tarifas más altas de lo normal. Mirando solo el resultado final seguramente no lo notaría, solo vería un error un poco más alto. Por eso hay que revisar los datos antes de entrenar.

## Ejercicio 1: resumen, distribución y atípicos

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv) tiene 3000 viajes en taxi de Nueva York: fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Primera mirada: qué contiene, cómo se distribuye la tarifa y qué valores están fuera de lo esperable.

**Respuesta 1, antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

1. El máximo de la tarifa es 999.99, que parece un valor puesto por error. Son 15 viajes.
2. El mínimo de la duración es -74 minutos, y un viaje no puede durar tiempo negativo. Son 10 viajes.

**Respuesta 2:** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No, no todos son errores. El IQR marcó 334 viajes con tarifa mayor a 60.40, pero solo 15 son los de 999.99. Para averiguarlo agrupé los atípicos por zona de origen y destino, y 320 de los 334 son viajes del aeropuerto JFK (177 salen de JFK y 143 llegan a JFK). Además, 302 cobran justo 70.00, que es la tarifa fija del aeropuerto, así que son tarifas reales.

**La formulación completa, tal como la resuelvo yo:**

Script: [`starter/EDA.py`](./starter/EDA.py), lo corro con `docker compose up --build` y las figuras se guardan en `datacontenedor/`.

- Reviso el dataset con `info()`, `isna()`, `duplicated()` y `describe()`. No hay duplicados, pero faltan 120 valores en `pasajeros` y 60 en `metodo_pago`.
- Hago un histograma y un boxplot de la tarifa. La mitad de los viajes está entre 14 y 33 USD. En el histograma se ven tres grupos: la mayoría de viajes entre 0 y 60, un grupo aparte de unos 300 viajes cerca de 70, y una barra sola en 1000 (los errores de 999.99). En el boxplot esos dos grupos aparecen como puntos fuera de los bigotes.
- Con el IQR saco los atípicos (334) y veo que la mayoría son viajes de JFK.
- Hago la dispersión de distancia contra tarifa sin los atípicos: la tarifa sube casi en línea recta con la distancia, pero hay una columna de viajes con distancia 0 y tarifas de hasta unos 30 USD, que reviso en el Ejercicio 2.
- Errores confirmados que hay que quitar: las 15 tarifas de 999.99 y las 10 duraciones negativas. Los demás atípicos se quedan porque son viajes reales.

## Durante el ConcepTest

_No se realizó ConcepTest en esta sesión._

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/explorar_relaciones.py`](./plantilla-ejercicio2/explorar_relaciones.py) y córrela desde su carpeta (`cd plantilla-ejercicio2`, `python3 explorar_relaciones.py`): lee el mismo `starter/viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

Después de quitar los errores quedan 2975 viajes. La tarifa sube casi en línea recta con la distancia (correlación de 0.96), pero hay dos grupos que no siguen esa tendencia:

- Una línea horizontal en 70 USD: son 300 viajes y todos son de JFK (165 salen y 135 llegan al aeropuerto). Es la tarifa fija, así que ahí la distancia no importa.
- Una columna en distancia 0: son 25 viajes con duración promedio de 27 minutos y tarifa promedio de 28.71 USD. El viaje sí pasó, pero la distancia no se registró bien.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

Los 888 viajes pagados en efectivo tienen propina 0, mientras que con tarjeta ningún viaje tiene propina 0 y el promedio es 4.73 USD. Creo que esto describe al sistema y no a los pasajeros: la propina en efectivo se le da en mano al conductor y no queda registrada. Además, hay 60 viajes sin método de pago.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La duración tiene una correlación de 0.96 con la tarifa, pero solo se conoce cuando el viaje ya terminó. Si alguien pide el taxi todavía no sabemos cuánto va a durar, así que el modelo no la puede usar. Lo mismo pasa con la propina. También vi que hay 119 viajes con pasajeros vacío y 19 con pasajeros en 0, que habría que revisar antes de usar esa columna.

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar, y la plantilla [`plantilla-eda/explorar_proyecto.py`](./plantilla-eda/explorar_proyecto.py).

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3.

Mi dataset es `obt_sales.csv.gz` (ventas diarias de Walmart por producto y tienda) y el objetivo es `UNITS`, las unidades vendidas en un día. El archivo tiene 59,181,090 filas y 40 columnas, y pesa 13 GB descomprimido. Por eso lo leo por partes: cuento los totales sobre todo el archivo y uso una muestra al azar del 1% (591,811 filas) para los gráficos. Para comprobar que la muestra representa bien al archivo, comparé dos porcentajes: los días con 0 unidades son 68.0% en el archivo completo y 68.0% en la muestra, y los días antes del lanzamiento son 20.8% en los dos.

- Faltantes: solo `SELL_PRICE` y `REVENUE`, con 12,299,413 vacíos cada una (20.8%).
- El 68% de las filas tiene `UNITS = 0`.
- En `figuras/distribucion_objetivo.png` (sin los días antes del lanzamiento): con un histograma normal todo caía en la primera barra, así que lo separé en dos. A la izquierda, barras de 0 a 20 unidades: el 0 es por mucho el valor más común y después cada valor tiene menos días. A la derecha, el eje y en escala logarítmica muestra una cola larga con pocos días de más de 100 unidades y uno de 544.
- El IQR marcó 70,346 atípicos en la muestra (11.9%). Como Q1 = 0 y Q3 = 1, cualquier día con 3 unidades o más ya cuenta como atípico.

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

Hice un gráfico de barras con las unidades promedio por día de la semana (`figuras/unidades_por_dia.png`). El fin de semana se vende más: sábado 1.36 y domingo 1.35 unidades, contra 0.97 el miércoles, que es el día más bajo (unos 40% más). También comparé por categoría (FOODS 1.65, HOUSEHOLD 0.73, HOBBIES 0.57) y por días con cupones SNAP (1.22 contra 1.09 sin SNAP).

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).

**Hallazgo 1:**

12,299,413 filas (20.8%) son días antes de que el producto empezara a venderse (`IS_PRE_LAUNCH`). En todas esas filas `UNITS` es 0 y el precio está vacío, así que no son ventas en cero reales, sino días en que el producto todavía no existía en la tienda. Hay que quitarlas antes de entrenar, porque si no el modelo aprende a predecir ceros de más.

**Hallazgo 2:**

Las ventas están muy concentradas en pocos valores: la mediana es 0, el promedio es 1.13 y el máximo es 544. Incluso sin contar los días antes del lanzamiento, cerca del 60% de los días tienen 0 unidades. Además, 1,946,313 días (3.3%) tienen sospecha de falta de stock, o sea que el cero es porque no había producto. El IQR no sirve para quitar atípicos aquí, porque marca como atípico cualquier día con 3 unidades o más. En el modelo hay que transformar el objetivo (por ejemplo con logaritmo) o usar un modelo para conteos, y revisar los días sin stock.

**Hallazgo 3:**

`REVENUE` tiene una correlación de 0.71 con `UNITS`, pero no se puede usar, porque se calcula como `UNITS × SELL_PRICE`: solo se conoce después de vender. Es lo mismo que pasaba con la duración en los taxis. `SELL_PRICE` sí se puede usar, porque el precio se conoce antes de vender (su correlación con `UNITS` es -0.15).

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Creando columnas nuevas a partir de las que ya existen:

- JFK: una columna `es_jfk` que vale 1 si el viaje sale o llega al aeropuerto y 0 si no. Así el modelo aprende que esos viajes cuestan 70 sin importar la distancia, que es la línea horizontal del Ejercicio 2.
- Hora del día: sacarla de `fecha_hora_inicio` (por ejemplo con `pd.to_datetime(...).dt.hour`). También se puede agrupar con menos detalle, como hora pico o no, o madrugada, día y noche, que es la idea de granularidad del artículo. Igual con el día de la semana o si es fin de semana.
