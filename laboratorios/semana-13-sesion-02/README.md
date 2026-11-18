# Semana 13, Sesión 2: Monitoreo en producción I: métricas de rendimiento

> Objetivo de la sesión: definir las métricas de monitoreo del modelo de tu proyecto en producción, distinguiendo métricas de sistema (latencia, disponibilidad) de métricas de modelo (desempeño).

**Lectura previa de esta sesión:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 8: Data Distribution Shifts and Monitoring.

## Requisitos antes de esta sesión

Nada nuevo que instalar. Los laboratorios de hoy son scripts. El Ejercicio 1 no necesita levantar uvicorn: `simular_noviembre.py` le manda las peticiones al servicio con `TestClient`, como las pruebas.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿por qué no alcanza con vigilar la salud y el volumen de peticiones del servicio? ¿Qué hay que medir además?**

Porque un servicio puede estar perfectamente sano y aun así dar predicciones malas. El artículo lo dice así: "no basta con seguir la salud y el volumen de peticiones del servicio desplegado". La latencia, los errores y el volumen dicen si el servicio **responde**, no si **acierta**. Un modelo se degrada en silencio: cambian los datos o la relación entre los datos y el objetivo, y sigue respondiendo 200 en milisegundos. Además del sistema, hay que medir:

- **El desempeño del modelo:** comparar las predicciones con la verdad que llega después (precisión, recall o RMSE, según el problema), para saber cuándo reentrenar.
- **La deriva:** de los datos de entrada (*data drift*), de las predicciones (*prediction drift*) y de la relación entre entrada y objetivo (*concept drift*).
- **La calidad de los datos de entrada:** cambios de esquema, features faltantes o inválidas, y diferencias entre cómo se calculan las features al entrenar y al servir (*training-serving skew*).
- **Los indicadores del negocio** a los que el modelo debería aportar.

**El artículo dice que la verdad (ground truth) a veces llega con demora, o no llega. ¿Qué propone medir mientras tanto?**

Propone vigilar señales que no necesitan la verdad y que suelen anticipar una caída del desempeño:

- **La deriva de las predicciones:** si la distribución de lo que predice el modelo cambia respecto de la del entrenamiento o de un período sano. Por ejemplo, la tasa de pagos enviados a revisión, o la tarifa promedio estimada. Si el modelo se comporta como en el entrenamiento, es probable que su calidad sea parecida.
- **La deriva de los datos y de las features:** si cambia la distribución de las entradas. Puede ser una primera señal de *concept drift*.
- **La calidad de los datos:** caídas en las predicciones exitosas y fallas de validación en el pipeline.

Además, propone relacionar las métricas del modelo con los indicadores del negocio. Ninguna de estas señales reemplaza a la verdad: avisan antes, pero pueden fallar en los dos sentidos (lo vimos en el Ejercicio 2).

## Pregunta de apertura

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Se degrada **en silencio**. Sigue respondiendo en milisegundos y con código 200, `/salud` dice `ok`, y cada tarifa trae su versión. Pero nada de eso mide si la tarifa acierta. Si cambian los datos, o la relación entre las features y el objetivo (por ejemplo, un aumento de tarifas), el modelo sigue aplicando lo que aprendió y se equivoca cada vez más, sin ningún error que lo delate. Para enterarme tengo que guardar cada predicción con un identificador, su entrada y su versión. Cuando llegue la verdad (la tarifa cobrada), la uno con la predicción y calculo el error por semana, comparado con el error con que se aprobó el modelo. Mientras la verdad no llega, vigilo la deriva de las entradas y de las predicciones, con alertas que me avisen antes que un usuario.

## Ejercicio 1: la tarifa, instrumentada

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa con la versión 1 del modelo (la red de 64 neuronas), ahora con dos agregados. Un middleware (una función que corre alrededor de cada petición) mide cuánto tarda cada predicción y con qué código responde; `/metricas` resume esas mediciones. Y cada predicción queda guardada en `predicciones.jsonl`, con un identificador que la app guarda junto al viaje. [`simular_noviembre.py`](./starter/ejercicio1/simular_noviembre.py) manda al servicio los 1156 viajes de noviembre y escribe `cobros.csv`, lo que el sistema de cobros anotó al terminar cada viaje: el identificador de la predicción y la tarifa cobrada (no lo modifiques). [`desempeno.py`](./starter/ejercicio1/desempeno.py) une las dos cosas y calcula el error por semana. Los 4 `TODO`: medir la latencia, resumir las métricas, guardar cada predicción y calcular el error por semana.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python simular_noviembre.py
python desempeno.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** los 1156 viajes recibieron su tarifa. ¿Qué dicen `/metricas` y `desempeno.py`? ¿Qué podrías afirmar sobre el servicio en noviembre?

`/metricas` devuelve `{}` (un diccionario vacío) y `desempeno.py` termina con `No hay predicciones.jsonl: el servicio no esta guardando sus predicciones.` Las dos pruebas fallan. Lo único que podría afirmar es que **el servicio respondió**: los 1156 viajes recibieron una tarifa y la simulación no se cayó. No podría decir cuánto tardó, cuántas peticiones fallaron ni, sobre todo, si las tarifas fueron buenas. Sin medir la latencia ni guardar las predicciones, no hay con qué saber si noviembre salió bien o mal.

**Respuesta 2 (TODO 1 y 2):** antes de correr la simulación, ¿qué latencia p95 esperas, y cuántos errores de validación? ¿Esperas que la latencia cambie entre la primera y la última semana del mes?

Esperaba una latencia p95 de **unos pocos milisegundos** (menos de 20 ms): el modelo es una red pequeña sobre una sola fila, y `TestClient` no pasa por la red. Esperaba **3 errores de validación**: `simular_noviembre.py` manda a propósito 3 peticiones sin fecha ni zona de destino, y 0 errores de servidor. Y **no** esperaba que la latencia cambie entre la primera y la última semana: el modelo, el código y el tamaño de cada petición son los mismos todo el mes. La latencia no sabe nada de si la tarifa acierta. Salió `p50 6.87 ms`, `p95 10.27 ms`, `errores_validacion 3`, `errores_servidor 0` en 1159 peticiones.

En clase hubo cuatro errores: faltaba el `await` en `call_next`, los contadores no empezaban en 0, los nombres de las variables y de las claves no coincidían con los que pide la prueba (`peticiones`, `latencia_p50_ms`, `latencia_p95_ms`), y se calculaban p95 y p99 en vez de p50 y p95. Los corregí con comentarios `#---------`.

**Respuesta 3 (TODO 3):** el servicio devuelve `id_prediccion` y lo guarda en el registro. ¿Para qué lo necesita el sistema de cobros? ¿Qué pasaría si solo se guardaran la fecha y la tarifa estimada?

Para **unir cada tarifa cobrada con la predicción exacta que se le mostró al pasajero**. El sistema de cobros no sabe nada del modelo: solo anota el `id_prediccion` que la app guardó junto al viaje y lo que se cobró. Con ese id, `desempeno.py` encuentra en el registro la tarifa estimada, la versión que la calculó y el viaje, y calcula el error. Si solo se guardaran la fecha y la tarifa estimada, no habría cómo saber qué cobro corresponde a qué predicción: en un mismo minuto puede haber varios viajes, un viaje se puede cotizar dos veces sin pedirse, y la fecha del cobro es la del final del viaje, no la de la cotización. Cualquier unión por fecha mezclaría viajes, y el error calculado no significaría nada. Por eso el registro guarda también la versión y la entrada completa: así el error se puede separar por versión y revisar viaje por viaje.

**Respuesta 4 (TODO 4):** el MAE en prueba de la versión 1 fue 1.17. Antes de correr `desempeno.py`, ¿qué esperas para cada semana de noviembre?

Esperaba un MAE **parecido a 1.17 en todas las semanas**, quizás algo más alto, porque los viajes de noviembre son nuevos para el modelo. Sin cambios en los viajes, el error de prueba debería repetirse en producción. No fue así: las dos primeras semanas se parecen a la prueba (1.18 y 1.34), pero desde la semana del 16 el error se multiplica por 3.7 (4.29 y 4.30).

**Lo que mostró el monitoreo:**

```
Metricas del sistema: {'peticiones': 1159, 'errores_validacion': 3, 'errores_servidor': 0, 'latencia_p50_ms': 6.87, 'latencia_p95_ms': 10.27}
```

| Semana | Viajes | MAE | Veces la referencia (1.17) |
|---|---|---|---|
| 2026-11-02 | 291 | 1.1792 | 1.01 |
| 2026-11-09 | 318 | 1.3403 | 1.15 |
| 2026-11-16 | 308 | **4.2916** | **3.67** |
| 2026-11-23 | 239 | **4.3042** | **3.68** |

La latencia p95 es de 10.27 ms, con 3 errores de validación (las peticiones mal formadas a propósito) y 0 errores de servidor. El sistema estuvo sano todo el mes, pero el modelo empeoró casi 4 veces desde el 16 de noviembre. Ninguna métrica de sistema lo mostró.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, cuando la verdad tarda tres semanas

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de fraude guardó los 5874 pagos que evaluó en noviembre en [`predicciones_fraude.csv`](./starter/ejercicio2/predicciones_fraude.csv): la probabilidad, si mandó el pago a revisión y cuánto tardó. Un fraude se confirma cuando el titular de la tarjeta reclama el cobro, entre 2 y 21 días después; los confirmados están en [`contracargos.csv`](./starter/ejercicio2/contracargos.csv). Hoy es 14 de diciembre. [`monitoreo_fraude.py`](./starter/ejercicio2/monitoreo_fraude.py) arma el reporte del mes. Los pasos 1 a 3 tienen `TODO`: la latencia p95 por día, la tasa de alertas por semana y los fraudes confirmados hasta hoy. Los pasos 4 (el recall de las semanas cuyos fraudes ya se confirmaron) y 5 (las alertas) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python monitoreo_fraude.py`.

**Respuesta 1:** el reporte da tres tipos de alerta. ¿Cuáles son de sistema y cuáles de modelo? ¿El día de la latencia alta coincide con alguna semana de recall bajo?

Son tres tipos:
- **De sistema:** `sistema: el 2026-11-10 la latencia p95 fue 78.2 ms`, con un umbral de 50 ms. Los demás días no pasan de 6.5 ms.
- **De modelo, indirecta:** `la semana del 2026-11-16 mando a revision 0.05 de los pagos` y `la semana del 2026-11-23 mando a revision 0.046 de los pagos`, por debajo del 75% de la tasa de referencia de 0.0758.
- **De modelo, directa:** `la semana del 2026-11-16 detecto 0.276 de los fraudes`, por debajo del 75% del recall de referencia de 0.661.

El día de la latencia alta **no** coincide con una semana de recall bajo. El 10 de noviembre cae en la semana del 9, que tuvo un recall de **0.736**, normal; el recall bajo es de la semana del 16. Son problemas independientes: un servicio lento puede acertar, y uno rápido puede fallar. `comprobar.py` da los cinco pasos en `OK`.

**Respuesta 2:** la tasa de alertas se puede calcular el mismo día; el recall, tres semanas después. ¿Qué semana mostró primero el problema la tasa de alertas, y cuándo lo confirmó el recall? ¿Por qué la tasa de alertas no alcanza para decir que el modelo empeoró?

La tasa de alertas mostró el problema **primero en la semana del 16 de noviembre**: bajó a 0.0503, contra 0.0706 y 0.0685 de las dos anteriores. Se podía ver ese mismo domingo 22. El recall lo confirmó recién cuando la semana maduró: su domingo, 22 de noviembre, más los 21 días que puede tardar un contracargo, da el **13 de diciembre**. Desde ese día el reporte muestra 0.276. Fueron 3 semanas de diferencia.

La tasa de alertas no alcanza porque mide cuántos pagos mandó a revisión el modelo, no si eran fraudes. Puede bajar porque llegan menos fraudes (y el modelo acierta al marcar menos), por un cambio en la mezcla de pagos (más clientes conocidos, otro método de pago) o porque el modelo empeoró. También puede quedar estable mientras el modelo empeora, si marca la misma cantidad de pagos pero los equivocados. Es una alerta temprana para ir a mirar, no una prueba. Solo el recall, con los fraudes confirmados, dice que el modelo dejó de detectarlos.

**Respuesta 3:** ¿por qué el reporte no muestra el recall de la semana del 23 de noviembre, si ya hay fraudes confirmados de esa semana?

Porque esa semana todavía **no está madura**. Su domingo es el 29 de noviembre, y un fraude de ese día se puede confirmar hasta el 20 de diciembre (29 + 21 días). Al 14 de diciembre hay 66 fraudes confirmados de esa semana, pero en el archivo de contracargos terminan siendo 82: faltan reclamos por llegar. Si se calculara el recall con los 66 de hoy (26 detectados), daría un número sesgado. Los fraudes que faltan no entran en ningún lado, y no se sabe si el modelo los habría detectado o no. Un recall calculado antes de tiempo cambia día a día y puede dar una alarma falsa o una tranquilidad falsa. Por eso el reporte solo calcula el recall de semanas cuyos fraudes ya pudieron confirmarse todos.

**Si terminas antes:** cambia `HOY` a `"2026-11-30"` y vuelve a correr. ¿Qué semanas tienen recall? ¿Desde qué día el reporte puede mostrar la caída del recall de la semana del 16?

Con `HOY = "2026-11-30"` solo tiene recall la **semana del 2 de noviembre** (0.767, 73 fraudes, 56 detectados): su domingo, el 8, más 21 días da el 29 de noviembre. La del 9 madura recién el 6 de diciembre. Las alertas indirectas de la tasa de alertas para las semanas del 16 y del 23 ya aparecen ese día; la del recall no. La caída del recall de la semana del 16 se puede mostrar **desde el 13 de diciembre** (domingo 22 de noviembre + 21 días). Lo comprobé: con `HOY = "2026-12-12"` el reporte tiene las semanas del 2 y del 9, y con `"2026-12-13"` aparece la del 16 con 0.276. Después devolví `HOY` a `"2026-12-14"`.

## Ejercicio 3: el monitoreo de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio de las dos sesiones anteriores. Esta vez no hay plantilla.

### Entregable 1: el servicio, instrumentado

- Un middleware que mida la latencia y el código de cada predicción, y una ruta `/metricas` con, al menos, la cantidad de peticiones, los errores de validación, los errores de servidor y la latencia p50 y p95.
- Cada predicción queda en un registro (un archivo JSONL en `monitoreo/`, por ejemplo) con su identificador, el momento, la versión del modelo, la entrada y la salida. Lo que haga falta para unirla después con la verdad tiene que estar en ese registro.
- Pruebas en `servicio/test_api.py` que comprueben que una predicción queda registrada y que `/metricas` separa los errores de validación de los de servidor, con un registro temporal (`tmp_path`).

### Entregable 2: `monitoreo/desempeno.py`

Une el registro con la verdad de tu proyecto y calcula la métrica de la Semana 11 por período (día o semana) y por versión. Si en tu proyecto la verdad no llega, o llega muy tarde, define una métrica que se pueda calcular antes y explica qué te dice y qué no.

### Entregable 3: `docs/monitoreo.md`

Una tabla con cada métrica que vigilas: si es de sistema o de modelo, de dónde sale, cuándo se conoce y con qué umbral alerta.

**¿Cuánto tarda en llegar la verdad en tu proyecto, y qué métrica te avisa mientras tanto?**

En mi proyecto la verdad llega **rápido**: las unidades vendidas de un producto en una tienda se conocen al cierre de ese día. Para la predicción del lote nocturno (la de mañana), la verdad llega en **unas 24 a 48 horas**. Para el horizonte de 28 días del M5, una predicción hecha hoy para dentro de 4 semanas se verifica en 28 días. `monitoreo/desempeno.py` une el registro (`monitoreo/predicciones.jsonl`) con las ventas reales por `(fecha, item_id, store_id)` y calcula el **costo de stock** (la métrica de la Semana 11) y el MAE por semana y por versión.

Mientras tanto, me avisan dos métricas que se calculan el mismo día, sin la verdad. La **escala de las predicciones** (unidades estimadas promedio por día-producto, comparada con 1.44, el promedio vendido en entrenamiento) muestra deriva de las predicciones. La **fracción de series sin referencias** (`serie_conocida = false`) muestra productos nuevos con features imputadas. En la simulación de 8 semanas de 2016 (3200 predicciones, versión 1 las primeras 4 semanas y versión 2 las últimas 4), el costo de la versión 1 estuvo entre 0.87 y 1.08 veces su referencia de 3.0457. La semana del 9 de mayo, con la versión 2, llegó a 4.11 (1.35 veces). Ese aviso solo lo dio la verdad: el promedio estimado (1.51) estaba dentro de lo normal, pero esa semana se vendió mucho más (1.88). La tabla completa de métricas y umbrales está en [`psets/proyecto/docs/monitoreo.md`](../../psets/proyecto/docs/monitoreo.md).

**¿Qué falla de tu modelo no vería ninguna de tus métricas de sistema?**

Que el modelo **subestime la demanda**, que es el error caro: una unidad que falta cuesta 3 veces una que sobra. Si llega una semana con más ventas de lo normal (como la del 9 de mayo de 2016), una promoción, un feriado que no estaba en el calendario o un cambio de hábitos, el servicio sigue respondiendo en 36 a 44 ms, sin errores de servidor, y `/salud` dice `ok`. Pero las tiendas piden menos de lo que van a vender y se quedan sin stock. Tampoco verían que mi modelo, sin `ITEM_ID`, subestima mucho los productos que más se venden (para `FOODS_3_090` estima 7.45 unidades cuando se vendieron 47), ni una feature mal calculada en el servicio, ni que un producto nuevo quede con sus features imputadas. Todo eso solo aparece al comparar con las ventas reales (el costo de stock por semana) o al mirar la escala de las predicciones y las series desconocidas.

## Antes de la próxima sesión

**Lectura:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe el servicio son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

Cambió **la relación entre los viajes y la tarifa**, no los viajes. Desde el 16 de noviembre se cobra alrededor de un **19% más** por el mismo viaje: la mediana de tarifa cobrada sobre tarifa estimada pasó de 1.00 a 1.19. Las distancias promedio (de 7.8 a 8.8 km), los pasajeros y las zonas de origen no cambiaron. Parece un aumento de tarifas: el modelo sigue cobrando lo que aprendió en octubre, y la realidad cambió de regla. Es **concept drift**: cambia P(tarifa | viaje), no la distribución de los viajes (que sería *data drift*).

**No lo hubiera detectado mirando solo los viajes que llegan**: su distribución es la misma, así que cualquier monitor de deriva de las entradas habría dicho que todo estaba bien. Tampoco mirando las predicciones, porque el modelo estima lo mismo que antes para los mismos viajes (unos 25 a 27 dólares de promedio). Solo se ve al unir las predicciones con la tarifa cobrada, que es lo que hizo `desempeno.py`: el MAE pasó de 1.18 a 4.29.
