# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el pipeline de tu proyecto de la Semana 9, Sesión 2.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje combina las predicciones de varios modelos (aprendices base) para obtener una sola predicción mejor que la de cualquiera de ellos. En **bagging** los modelos se entrenan en paralelo e independientes, cada uno sobre una muestra con reemplazo (bootstrap) de los datos, y al final se promedian (regresión) o se vota (clasificación); sobre todo baja la varianza. En **boosting** los modelos se entrenan en secuencia: cada uno nuevo se enfoca en los errores que dejaron los anteriores, y la predicción es una suma ponderada; sobre todo baja el sesgo.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Porque cada árbol se equivoca de forma distinta: al entrenarse con muestras y columnas diferentes, sus errores son en parte independientes, y al promediarlos los errores hacia arriba y hacia abajo se cancelan. El ruido (la varianza) baja, mientras que lo que todos aprendieron bien (la señal) se mantiene. El mejor árbol solo, en cambio, conserva todo su ruido propio.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está sobreajustando: memorizó los datos de entrenamiento, incluido el ruido, y no generaliza a datos nuevos. Tiene mucha varianza. Para elegir entre los dos hay que fijarse en el error en prueba, no en el de entrenamiento.

## Ejercicio 1: línea base, un árbol y un bosque

Archivo: [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. El script ya trae la preparación de la Semana 9 y una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-01/starter
python3 ejercicio1_linea_base_y_ensamblaje.py
```

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Espero que se equivoque más o menos lo que se alejan las tarifas de su mediana, unos 12 a 13 dólares por viaje (las tarifas van de 2 a 78 dólares, con mediana de unos 22). Un modelo que no usa ninguna columna sirve de piso: cualquier modelo que valga la pena tiene que equivocarse bastante menos que él. Si no lo supera, el modelo no aprendió nada útil de las columnas.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

En entrenamiento espero un error de casi 0, porque con una hoja por viaje el árbol se aprende de memoria cada tarifa. En prueba espero un error bastante mayor, porque esos cortes tan finos copian el ruido de entrenamiento y no sirven para viajes nuevos: una brecha grande.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad a 8, el error en entrenamiento sube (ya no puede memorizar) pero la brecha se achica y el error en prueba debería bajar un poco. Con 200 árboles promediados espero un error en entrenamiento bajo, pero no 0, y una brecha menor que la del árbol sin límite, porque el promedio cancela parte del ruido de cada árbol.

**Lo que confirmó la tabla, TODO por TODO:**

| Modelo | Entrenamiento | Prueba | Brecha | Segundos |
|---|---|---|---|---|
| línea base | 13.1998 | 12.5308 | -0.6690 | 0.011 |
| árbol sin límite | 0.0029 | 1.7017 | 1.6988 | 0.020 |
| árbol max_depth=8 | 1.0098 | 1.2728 | 0.2630 | 0.013 |
| bosque aleatorio | 0.4739 | 1.3016 | 0.8276 | 1.511 |

- TODO 1: la línea base se equivoca 12.53 dólares en prueba, como esperaba. Es el piso.
- TODO 2: el árbol sin límite casi no se equivoca en entrenamiento (0.003) pero en prueba se equivoca 1.70: se sobreajusta, con la brecha más grande (1.70).
- TODO 3: con `max_depth=8` el error en entrenamiento sube a 1.01, pero en prueba baja a 1.27 y la brecha se reduce a 0.26. Es el mejor en prueba.
- TODO 4: el bosque baja mucho la brecha frente al árbol sin límite (de 1.70 a 0.83) y mejora la prueba (1.30), pero no le gana al árbol limitado: sus árboles siguen sin límite, así que cada uno memoriza, y además tarda unas 100 veces más en entrenar.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(se completa en clase: el enunciado del ConcepTest no está en el repositorio)_

**Tu razonamiento:**

_(se completa en clase)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(se completa en clase)_

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Archivo: [`starter/ejercicio2_regularizacion.py`](./starter/ejercicio2_regularizacion.py), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_regularizacion.py
```

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

Recibe 3275 columnas y hay solo 2380 filas de entrenamiento: hay más columnas (coeficientes) que filas. Con eso la regresión lineal tiene libertad para pasar exactamente por casi todos los puntos de entrenamiento (error 0.89), pero los coeficientes se disparan a valores enormes, y en viajes nuevos las predicciones explotan: el error en prueba fue de 1,695,287,900 dólares. Es sobreajuste extremo.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

En entrenamiento el error subió un poco, de 0.8922 a 1.0272. En prueba bajó de 1,695,287,900 a 1.3209, y la brecha pasó de unos 1.7 mil millones a 0.29. Ridge empeora en entrenamiento porque castiga los coeficientes grandes (`alpha=10`): ya no puede usar las 3275 columnas con total libertad para ajustar cada punto, así que acepta equivocarse un poco más en entrenamiento a cambio de coeficientes pequeños y estables que sí sirven en datos nuevos.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

El mejor en prueba fue `gradient boosting`, con 1.2754 dólares de error (brecha 0.20, la más chica de los modelos que aprenden). La línea base se equivoca 12.5308, así que el boosting se equivoca 11.26 dólares menos por viaje, un 89.8% menos.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

Porque la línea base no aprende de las columnas, solo calcula la mediana de entrenamiento (22.53). Su error depende solo de qué tan dispersas están las tarifas de cada conjunto alrededor de ese número, y por azar de la división las tarifas de prueba están un poco menos dispersas (desviación estándar 17.32 contra 18.17 en entrenamiento). La brecha negativa no es mérito del modelo, es ruido de la división; en un modelo sin capacidad de memorizar, la brecha debe salir cerca de 0, hacia un lado u otro.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el script tú.

### Entregable 1: `modelos_proyecto.py`

Un script, en la carpeta de tu proyecto, que entrene con tu pipeline:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

`DummyRegressor(strategy="median")`, que predice siempre la mediana de `UNITS` de entrenamiento. Mi métrica es el error absoluto medio (MAE), y la constante que menos MAE da es justamente la mediana. Como el 59.8% de los días de entrenamiento tienen 0 unidades, la mediana es 0: la línea base es "este producto no vende nada hoy", que es difícil de superar con MAE. Comparo también contra el promedio, porque la regresión lineal y los árboles con error cuadrático aprenden promedios, no medianas. No uso "mañana igual que hoy" aunque sea una serie en el tiempo, porque mi muestra es un 1% al azar de las filas: casi nunca está el día anterior del mismo producto en la misma tienda.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

Sí, el árbol sin límite: se equivoca 0.0186 unidades en entrenamiento y 2.1730 en prueba, una brecha de 2.15. Memorizó cada fila y en prueba es el peor de todos, peor incluso que las dos líneas base. Los demás no se sobreajustan; su brecha incluso sale un poco negativa, porque 2016 tiene algo menos de ventas que el entrenamiento:

| Modelo | MAE entrenamiento | MAE prueba | Brecha | Segundos |
|---|---|---|---|---|
| línea base (mediana) | 1.4370 | 1.4103 | -0.0267 | 0.6 |
| línea base (promedio) | 1.8648 | 1.7677 | -0.0971 | 0.6 |
| lineal | 1.8120 | 1.7238 | -0.0883 | 0.8 |
| ridge alpha=10 | 1.8098 | 1.7202 | -0.0896 | 0.7 |
| árbol sin límite | 0.0186 | 2.1730 | 2.1545 | 3.7 |
| bosque aleatorio | 1.5368 | 1.5069 | -0.0299 | 74.0 |
| gradient boosting | 1.6487 | 1.5659 | -0.0829 | 71.6 |
| gradient boosting (absoluto) | 1.4370 | 1.4103 | -0.0267 | 3.2 |

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

En el modelo lineal comparé `LinearRegression` contra `Ridge(alpha=10)` y casi no cambió nada: el MAE en prueba pasó de 1.7238 a 1.7202. No hacía falta, porque con 26 columnas y 424,975 filas hay muchísimas más filas que coeficientes (al revés que en el Ejercicio 2, con 3275 columnas y 2380 filas). Donde sí hizo falta fue en los árboles: el árbol sin límite tiene una brecha de 2.15, y el bosque con `max_depth=12` y `min_samples_leaf=50` la baja a -0.03 y el MAE en prueba de 2.17 a 1.51.

**¿Cuánto mejora tu mejor modelo a la línea base?**

Contra la línea base de la mediana, no mejora nada: ningún modelo le gana. La mediana (predecir siempre 0) se equivoca 1.4103 unidades en prueba, y el mejor modelo que aprende, el bosque, se equivoca 1.5069, un 6.8% más. El `gradient boosting (absoluto)` empata exactamente con la línea base porque, con un 60% de ceros, la mediana de los residuos también es 0 en casi todas las hojas, y termina prediciendo 0 siempre. Los otros modelos se entrenan con error cuadrático y aprenden promedios, que con MAE pierden ante la mediana cuando hay tantos ceros.

Contra la línea base del promedio, que es la comparación justa para modelos que aprenden promedios, el bosque sí mejora: de 1.7677 a 1.5069, un 14.8% menos de error. O sea, las columnas sí tienen información, pero no basta para superar "hoy no vende nada" con MAE. Para la próxima sesión hay que revisar la métrica (por ejemplo RMSE, o el MAE solo en los días con venta) y agregar features por producto, como sus ventas promedio, porque hoy el modelo no sabe qué producto es (dejé fuera `ITEM_ID`).

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error, mediría:

- **Latencia:** cuánto tarda cada predicción. La app necesita mostrar la tarifa en milisegundos; un ensamblaje responde en milisegundos, un LLM puede tardar uno o varios segundos.
- **Costo por predicción:** el ensamblaje corre casi gratis en un servidor propio; un LLM cobra por token en cada viaje, y con millones de viajes eso suma.
- **Consistencia:** el mismo viaje tiene que dar siempre la misma tarifa. El ensamblaje es determinista; el LLM puede devolver números distintos o hasta texto que no es un número.
- **Costo y tiempo de entrenamiento y de mantenimiento:** el ensamblaje entrena en segundos con mis datos, la red neuronal necesita más datos y más ajuste, y el LLM no se entrena con mis datos pero depende de un proveedor externo.
- **Explicabilidad:** poder decir por qué una tarifa salió alta (distancia, hora pico), algo que pide el negocio y a veces la regulación.
- **Privacidad:** mandar los datos de cada viaje a un servicio externo.
