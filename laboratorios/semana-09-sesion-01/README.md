# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tus 3 hallazgos de la exploración de la Semana 8, Sesión 2: las features de hoy salen de ahí.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature con sentido de negocio representa un comportamiento real que se puede explicar, y la gente del negocio la entiende. Una que solo mejora la métrica puede salir de sumar o multiplicar columnas al azar: quizás predice un poco mejor, pero nadie sabe qué significa, y así es difícil convencer a alguien de usar el modelo. El artículo propone tres formas de crear features con sentido: ratios, tendencias y granularidad.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

El de las compras en línea (ratios). Si dos clientes hicieron 5 compras en línea, con el número solo parecen iguales. Pero si uno hizo 100 compras en total y el otro solo 5, el primero compra en línea el 5% de las veces y el segundo el 100%. El conocimiento del negocio es que lo que importa es qué tan "digital" es el cliente, y eso se traduce en una columna nueva: compras en línea dividido para compras totales.

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Creando columnas nuevas a partir de las que ya existen. Para la tarifa fija, una columna `es_tarifa_fija` que vale 1 si el viaje va entre JFK y Manhattan. Para las horas, sacar la hora de `fecha_hora_inicio` y crear `es_hora_pico`, que vale 1 en las horas con tráfico. Así el modelo ve los dos patrones como columnas.

## Ejercicio 1: features de fecha y hora

Archivo: [`starter/ejercicio1_features_tiempo.py`](./starter/ejercicio1_features_tiempo.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. El script ya quita los errores confirmados, convierte `fecha_hora_inicio` y compara la tarifa por kilómetro según la hora pico; faltan 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-01/starter
python3 ejercicio1_features_tiempo.py
```

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como número, la hora 23 y la hora 0 quedan a 23 de distancia, lo más lejos posible. Pero en el reloj solo hay 1 hora entre las dos. El modelo pensaría que las 11 de la noche y la medianoche son momentos muy distintos, cuando en realidad son casi lo mismo.

**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?

Más alta. En hora pico hay tráfico, el taxi avanza más lento y cada kilómetro toma más minutos. Como la tarifa también cobra por tiempo, cada kilómetro sale más caro. La salida lo confirma: 3.52 USD por km en hora pico contra 3.22 fuera de ella, y la velocidad baja de 21.48 a 16.51 km/h.

**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

La parte del tráfico. La duración le diría al modelo si el viaje fue lento, y `es_hora_pico` le avisa que es probable que lo sea, porque en esas horas se va más lento. No da la duración exacta, pero sí la idea de "este viaje va a tardar más por kilómetro", y eso sí se sabe al pedir el taxi.

**Lo que confirmó la salida, TODO por TODO:**

- TODO 1 y 2: el primer viaje (2026-09-06 13:46) queda con hora 13 y día 6, que es domingo, así que `es_fin_de_semana` = 1.
- TODO 3: tarifa por km (mediana) de 3.22 USD fuera de hora pico y 3.52 USD en hora pico.
- TODO 4, Paso 5: 23 y 0 quedan a 23 como número y a 0.26 como ciclo; 11 y 12 quedan a 1 como número y también a 0.26 como ciclo. Con seno y coseno, dos horas seguidas quedan siempre a la misma distancia, incluso al pasar de las 23 a las 0.

## Durante el ConcepTest

_No se realizó ConcepTest en esta sesión._

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Archivo: [`starter/ejercicio2_features_zona.py`](./starter/ejercicio2_features_zona.py), sobre el mismo `viajes_taxi.csv`. Completa los 4 `TODO`. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_features_zona.py
```

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

300 viajes tienen `es_tarifa_fija` en 1, y todos cuestan exactamente 70 USD (mínimo 70 y máximo 70). Con esta columna, un modelo lineal aprende a sumar o restar una cantidad fija para esos viajes, pero igual les sigue sumando lo que cobra por kilómetro, así que para un viaje largo a JFK predeciría más de 70. Para que aprenda que ahí la distancia no importa, haría falta combinar la columna con la distancia (por ejemplo multiplicarlas) o usar un modelo de árboles, que sí puede separar esos viajes.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea 8 columnas, una por zona, y la ordinal crea 1 sola, con números del 0 (Brooklyn) al 7 (Upper West Side). Para una regresión lineal usaría one-hot. La ordinal le dice al modelo que las zonas tienen un orden y distancias entre ellas, por ejemplo que Upper West Side (7) es "más" que Midtown (4) y que Harlem (1) está más cerca de Brooklyn (0) que Queens (5). Ese orden solo sale del alfabeto y no significa nada.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

`duracion_min` y `propina`, que solo se conocen al terminar el viaje, y `metodo_pago`, que se decide al pagar. La `tarifa` tampoco, porque es lo que queremos predecir. Si las usara, el modelo saldría muy bien en el entrenamiento porque estaría "viendo el futuro" (data leakage), pero al usarlo de verdad esas columnas no existirían y no podría predecir.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de la exploración de la Semana 8 y la plantilla [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py).

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

Mi modelo predice `UNITS`, cuántas unidades de un producto se van a vender en una tienda en un día. La predicción se hace antes de que empiece ese día, para saber cuánto stock tener.

- **Existe en ese momento:** la fecha y todo el calendario (día de la semana, mes, eventos, días SNAP), el producto, la tienda, la categoría, el precio (Walmart lo fija por semana, así que se sabe antes) y las ventas de los días anteriores.
- **Todavía no existe:** las `UNITS` de ese día, `REVENUE` (que es `UNITS × SELL_PRICE`) y, probablemente, `IS_SALES_SPIKE` e `IS_SUSPECTED_STOCKOUT_...`, que mi pipeline marca mirando las ventas de ese mismo día.

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

El archivo tiene 59 millones de filas, así que la plantilla lo lee por partes (igual que en la Semana 8). Antes de calcular nada, separo entrenamiento (hasta 2015) y prueba (2016), porque el precio promedio de cada producto lo calculo solo con entrenamiento. Las dos features se miden en una muestra al azar del 1% y solo con los días de entrenamiento (424,975 filas).

```python
# Feature 1: cuantos dias lleva el producto a la venta en esa tienda
datos["dias_desde_lanzamiento"] = (pd.to_datetime(datos["DATE"]) - pd.to_datetime(datos["fecha_lanzamiento"])).dt.days

# Feature 2: precio del dia dividido para el precio promedio del producto (calculado solo con entrenamiento)
datos["precio_relativo"] = datos["SELL_PRICE"] / datos["precio_promedio"]
```

`fecha_lanzamiento` es el primer día en que cada producto se vendió en cada tienda (el primer día sin `IS_PRE_LAUNCH`), y `precio_promedio` es el promedio de `SELL_PRICE` de cada producto en cada tienda antes de 2016.

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

`dias_desde_lanzamiento`. Sale de mi Hallazgo 1 de la Semana 8: el 20.8% de las filas son días antes de que el producto empezara a venderse, así que cada producto tiene una fecha de lanzamiento. La idea es que un producto nuevo vende distinto a uno que lleva años en la tienda (la "tendencia" del artículo).

La evidencia es débil: la correlación con `UNITS` es -0.008 y el promedio casi no cambia con la antigüedad (1.46 en los primeros 30 días, 1.54 entre 31 y 90 días, 1.45 entre 91 y 365, y 1.43 después de un año). Solo se ve un pequeño aumento entre el primer y el tercer mes. Además, para los productos que ya existían cuando empiezan los datos (2011-01-29) la antigüedad real es mayor de lo que dice la columna. Aun así la dejaría para probarla en el modelo, porque un modelo de árboles puede encontrar efectos que el promedio general no muestra. Existe al momento de predecir porque la fecha de lanzamiento ya pasó y la fecha del día se conoce. Es numérica, así que no necesita codificación.

**Feature 2:**

`precio_relativo`. Sale del conocimiento del negocio y de la idea de "ratios" del artículo: el precio solo no dice si un producto está barato, porque un precio de 3 USD puede ser caro para un producto y barato para otro. Dividido para su precio promedio, 0.9 significa "10% más barato de lo normal", es decir, en oferta.

La evidencia es más clara: en oferta (menos de 0.95) se venden 1.94 unidades en promedio, contra 1.36 a precio normal, un 43% más. El efecto viene sobre todo de FOODS (2.77 en oferta contra 2.03 normal), mientras que en HOBBIES y HOUSEHOLD casi no cambia. La correlación lineal es baja (-0.022) porque la relación no es una línea recta: también los días con precio más alto (más de 1.05) venden más que los normales (1.67), seguramente porque los precios suben con los años y mezcla productos de distintos periodos. Por eso convendría un modelo de árboles o cruzar la feature con la categoría. Existe al momento de predecir porque el precio de la semana se fija antes, y el promedio se calcula solo con entrenamiento, así no hay data leakage. Es numérica, así que no necesita codificación.

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Si uso todos los viajes, la media ya incluye información de los viajes de prueba, y el modelo estaría aprendiendo algo de datos que supuestamente no ha visto. Eso es data leakage: según el artículo, hay que separar entrenamiento y prueba antes de transformar, ajustar (`fit`) solo con entrenamiento y después aplicar (`transform`) esa misma media a los dos. Con una sola media el cambio en los números puede ser pequeño, pero la evaluación sale más optimista de lo que en verdad es, porque en la vida real los viajes nuevos no estaban cuando se calculó la media.
