# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

scikit-learn ya viene en el Codespace. En tu laptop, instálalo con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. El `requirements.txt` del repositorio ya fija una versión posterior.

Ten a mano el script de features de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

Es cuando información de prueba, o información que no existe al momento de predecir, se mete en el entrenamiento. El modelo se ve excelente en la evaluación porque, sin saberlo, ya "vio" parte de las respuestas, como estudiar con el examen filtrado. Pero en producción esa información no está, y el modelo predice mucho peor de lo que prometía.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

La fuga del objetivo (target leakage): usar como entrada una columna que es consecuencia de lo que se quiere predecir y que no existe al momento de predecir. La duración solo se conoce cuando el viaje terminó, igual que el ejemplo del artículo de las llamadas a servicio al cliente, que solo pasan cuando el cliente ya decidió irse.

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Si uso todos, la media ya trae información de los viajes de prueba y hay fuga. Lo correcto es dividir primero, hacer `fit` solo con entrenamiento y después aplicar esa misma media a prueba con `transform`. En los números cambia poco (en el Ejercicio 1, la media de distancia que usa el escalador es 8.5034 con todos los viajes y 8.5576 solo con entrenamiento), pero la evaluación deja de ser honesta.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

En `X_preparado = escalador.fit_transform(imputador.fit_transform(X))`. Ahí el imputador y el escalador aprenden con todo `X`, y la división en entrenamiento y prueba viene después. Lo delata "Filas que vio el escalador: 2975": son todas las filas, cuando el entrenamiento debería tener solo el 80% (2380).

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

Mide qué tan bien el modelo repite datos que ya vio, no qué tan bien predice datos nuevos. Es como corregir un examen con las mismas preguntas que se usaron para estudiar: no dice nada de cómo le irá con viajes que nunca vio.

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Esperaba que cambiara poco, y casi no cambió: 2.5375 con fuga y 2.5373 sin fuga. Pero sí había fuga, porque el escalador vio las 2975 filas. Cambia poco porque las 595 filas de prueba se parecen mucho a las de entrenamiento, así que la mediana y la media casi no se mueven, y además una regresión lineal da las mismas predicciones aunque los datos estén escalados. Que el error no cambie no prueba que no haya fuga: con otros datos o con otro modelo podría pesar mucho más.

**Lo que confirmó la salida, TODO por TODO:**

- Versión del compañero: el escalador vio 2975 filas, media de distancia 8.5034, error en prueba 2.5375.
- TODO 1: ahora prueba tiene 595 filas y entrenamiento 2380.
- TODO 2 y 3: el escalador está dentro del pipeline y solo vio las 2380 filas de entrenamiento, con media de distancia 8.5576. Error en prueba: 2.5373.

## Durante el ConcepTest

_No se realizó ConcepTest en esta sesión._

## Ejercicio 2: el pipeline completo

Solo o en pareja. Archivo: [`starter/ejercicio2_pipeline.py`](./starter/ejercicio2_pipeline.py), sobre el mismo `viajes_features.csv`. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_pipeline.py
```

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

Quedan 25 columnas:

- Numéricas: 2 (`distancia_km` y `pasajeros`), que solo se imputan y escalan, así que siguen siendo 2.
- Categóricas: 18, porque one-hot crea una columna por cada valor: 8 zonas de origen, 8 zonas de destino y 2 métodos de pago (efectivo y tarjeta).
- Ya listas: 5 (`hora_sin`, `hora_cos`, `es_fin_de_semana`, `es_hora_pico` y `es_tarifa_fija`), que pasan sin cambios.

2 + 18 + 5 = 25.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con las de entrenamiento: el imputador aprendió una distancia de 6.94, igual que la mediana de entrenamiento, mientras que la de todo el dataset es 6.93 (en pasajeros las tres dan 1.0). Eso demuestra que el pipeline aprendió solo con entrenamiento y que los datos de prueba no se metieron.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

Como el Bronx nunca apareció en entrenamiento, con `handle_unknown="ignore"` las 8 columnas de zona de origen quedan en 0, y el viaje igual sale con sus 25 columnas, así que el modelo puede usarlo. Con `pd.get_dummies` aplicado solo a ese viaje habrían salido 10 columnas distintas, entre ellas una `zona_origen_Bronx` que el modelo no conoce, y faltarían todas las demás. No coincidiría con las columnas de entrenamiento y el modelo no podría predecir.

## Ejercicio 3: el pipeline de tu proyecto

Usa tu archivo con las 2 features de la sesión pasada y la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py).

### Paso 1: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

Uso `obt_sales.csv.gz` (ventas diarias de Walmart), a través de la muestra del 1% con mis 2 features que guarda el script de la sesión pasada (468,820 filas, sin los días antes del lanzamiento).

- **Objetivo:** `UNITS`, las unidades vendidas de un producto en una tienda en un día.
- **Numéricas** (se imputan con la mediana y se escalan): `SELL_PRICE`, `precio_relativo` y `dias_desde_lanzamiento`.
- **Categóricas** (one-hot): `CAT_ID` (3 categorías), `STORE_ID` (10 tiendas) y `WEEKDAY` (7 días).
- **Ya listas** (pasan sin cambios): `IS_WEEKEND`, `IS_EVENT_DAY` e `IS_SNAP`, que ya son 1 o 0.
- **Fuera, porque no existen al predecir:** `REVENUE` (es `UNITS × SELL_PRICE`), `IS_SALES_SPIKE` e `IS_SUSPECTED_STOCKOUT_...`, que se marcan mirando las ventas del mismo día. Lo comprobé en las primeras 300 mil filas: cuando `IS_SUSPECTED_STOCKOUT_CONSERVATIVE` es 1, `UNITS` siempre es 0, y cuando `IS_SALES_SPIKE` es 1, `UNITS` siempre es 21 o más. Le estarían dando la respuesta al modelo.
- **Fuera por otras razones:** `ITEM_ID`, porque tiene 3049 productos y one-hot crearía 3049 columnas, y `DATE`, que solo uso para dividir.

### Paso 2: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha; si no, al azar. Justifica.

Divido por fecha: entrenamiento hasta el 2015-12-31 y prueba desde el 2016-01-01 hasta el 2016-05-22, el mismo corte de la sesión pasada. Mi modelo predice ventas de días que todavía no pasaron. Si dividiera al azar, el modelo entrenaría con días de 2016 para predecir días de 2015, o sea que vería el futuro, y eso también es fuga. Con la división por fecha, la prueba se parece a usar el modelo de verdad.

### Paso 3: construye y corre el pipeline (15 minutos)

Completa la plantilla y córrela. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

- **Entrenamiento:** (424975, 26). **Prueba:** (43845, 26).
- **Las 26 columnas:** 3 numéricas + 20 de one-hot (3 categorías + 10 tiendas + 7 días) + 3 ya listas:
  - `SELL_PRICE`, `precio_relativo`, `dias_desde_lanzamiento`
  - `CAT_ID_FOODS`, `CAT_ID_HOBBIES`, `CAT_ID_HOUSEHOLD`
  - `STORE_ID_CA_1` a `CA_4`, `TX_1` a `TX_3`, `WI_1` a `WI_3`
  - `WEEKDAY_Monday` a `WEEKDAY_Sunday`
  - `IS_WEEKEND`, `IS_EVENT_DAY`, `IS_SNAP`
- **Vacíos:** había 21 en `precio_relativo`, todos en prueba. Son productos que salieron en 2016 y no tienen precio promedio de entrenamiento. El imputador los llenó con la mediana de entrenamiento y después de preparar quedan 0 vacíos.
- **Comprobación de que no hay fuga:** el imputador aprendió las medianas de entrenamiento (`SELL_PRICE` 3.44 y `dias_desde_lanzamiento` 729), no las de todo el dataset (3.47 y 790). En mi proyecto la diferencia es más grande que con los taxis: como los productos van envejeciendo, los datos de 2016 cambian bastante la mediana de antigüedad.

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Está sobreajustando (overfitting): se aprendió de memoria los datos de entrenamiento, incluido el ruido, y por eso no generaliza a datos nuevos. En los términos del artículo, tiene mucha varianza. El artículo explica que los ensembles, como random forest (bagging), ayudan justamente con esto: entrenan muchos modelos distintos y promedian sus predicciones, y así los errores de cada uno se compensan.
