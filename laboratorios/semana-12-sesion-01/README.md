# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

_(tu respuesta)_

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

_(tu respuesta)_

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La tarifa se podría calcular de noche, de forma aproximada: casi todo depende de la zona de origen, la de destino, la hora y el tipo de día, que son combinaciones limitadas, y se guardaría una tabla para consultarla al instante. El fraude no, porque depende de cosas que solo existen en el momento del pago, como el monto, si la tarjeta es nueva y los intentos fallidos. Ese modelo tiene que responder en línea y antes de aprobar el pago, en menos de un segundo aproximadamente, para que el pasajero no note la espera.

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

Porque esas dos pruebas no tenían ningún `assert`. Para pytest, una prueba pasa si termina sin lanzar un error, y como `predecir_lote` sin resolver no fallaba (solo armaba la tabla con `id_viaje`), las dos pasaban sin comprobar nada. La única que comprobaba algo era `test_columnas_de_salida`, y esa sí falló porque faltaban `tarifa_estimada` y `version_modelo`. Una prueba sin `assert` no prueba nada.

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Espero que les estime una tarifa. El pipeline guardado incluye la preparación, que imputa los vacíos con lo que aprendió al entrenar: la mediana de `pasajeros` y el método de pago más frecuente. Saltarlos dejaría 6 viajes reservados sin tarifa en la tabla que la app lee por la mañana, y fallar dejaría sin tarifa a los 60. Así pasó: los 4 viajes sin pasajeros recibieron 33.94, 29.88, 25.80 y 17.58, y los 2 sin método de pago 60.79 y 29.59. La prueba del `TODO 5` comprueba justamente que un viaje incompleto también recibe tarifa.

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Para la trazabilidad. Cuando el modelo se reentrena, el archivo cambia y su huella también; si mañana un pasajero reclama por una tarifa, o se juntan las tablas de varias noches, cada fila dice exactamente qué modelo la produjo. Así se puede comparar el modelo nuevo con el anterior, auditar un error o volver atrás, aunque la fila se haya separado de la tabla o del log de la corrida.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

La del `TODO 4` falla si el script pierde, duplica o reordena viajes: por ejemplo, si hiciera `viajes.dropna()` antes de predecir (se saltaría los viajes incompletos) o si ordenara la salida por tarifa. La del `TODO 5` falla si un viaje incompleto queda sin tarifa (por ejemplo, si se quitara la imputación y se pusiera `NaN` a los viajes con vacíos) o si la versión no fuera la que se pasa (por ejemplo, una versión escrita a mano como `"v1"`). Al revisar las pruebas que habíamos escrito en clase corregí dos cosas: la del `TODO 4` hacía `assert list(a == b)`, que siempre es verdadero porque una lista no vacía cuenta como verdadera aunque tenga `False` adentro; y la del `TODO 5` volvía a comparar los `id_viaje` en vez de comprobar la `version_modelo`.

**Lo que mostró el lote:**

Versión del modelo: `8cf725276958` en las 60 filas. La tarifa más baja fue **4.67** (viaje 90014, 0.4 km dentro del Upper East Side) y la más alta **70.26** (viaje 90051, del Upper East Side al aeropuerto JFK, con tarifa fija). Los 4 viajes sin pasajeros sí recibieron tarifa: 90023 = 33.94, 90024 = 29.88, 90025 = 25.80 y 90048 = 17.58 (y los 2 sin método de pago: 90044 = 60.79 y 90050 = 29.59). Las 3 pruebas pasan: `3 passed`.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

Quedaron **26 pagos** en la cola de revisión con el umbral de 0.10. En la Semana 11 había 295 fraudes en 6000 pagos (4.9%), así que en 300 pagos esperaría unos **15 fraudes**. La cola es más larga que los fraudes esperados a propósito: con el umbral de 0.10 se aceptan más falsas alarmas (3 dólares cada una) para dejar pasar menos fraudes (40 dólares cada uno).

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

Pierde la posibilidad de **detener el fraude antes de que ocurra**: cuando el lote corre, el pago de ayer ya se aprobó y el viaje ya se hizo, así que la empresa solo puede reaccionar después (bloquear la cuenta o la tarjeta, intentar recuperar el dinero) y, mientras tanto, la misma tarjeta robada puede seguir usándose todo el día. Para que eso importe, la revisión tendría que poder actuar sobre el pago antes de que se pierda el dinero: por ejemplo, retener el cobro o bloquear la tarjeta en minutos. Si el equipo de revisión de todas formas trabaja la cola al día siguiente, evaluar cada pago al instante no cambiaría mucho; entonces haría falta revisar en el momento, o bloquear automáticamente los pagos más sospechosos.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

Que prueba solo la lógica de la cola, sin depender del modelo. Con probabilidades escritas a mano se sabe de antemano la respuesta correcta (4, 2 y 3, en ese orden) y se pueden poner casos difíciles a propósito, como el 0.10 justo en el umbral, que comprueba que se usa `>=` y no `>`. Además la prueba es rápida, da siempre lo mismo, y corre en GitHub Actions, donde no existe `modelo_fraude.joblib`. Si falla, se sabe que el error está en `a_revisar` y no en el modelo ni en los datos.

**Si terminas antes:** sube `UMBRAL` a 0.50 y vuelve a correr `puntuar_pagos.py`. ¿Cuántos pagos quedan en la cola? ¿Siguen pasando todas las pruebas? ¿Por qué?

Con `UMBRAL = 0.50` quedan **13 pagos** en la cola (contra 26 con 0.10), y siguen pasando las 4 pruebas (`4 passed`). Pasan porque ninguna prueba usa la constante `UMBRAL`: la prueba de la cola le pasa su propio umbral de 0.10 a `a_revisar`, y las demás no dependen del umbral. Las pruebas comprueban que el código hace bien lo que hace con cualquier umbral; qué umbral conviene al negocio es una decisión que se tomó con el costo en la Semana 11, no algo que pruebe pytest. Después lo devolví a 0.10.

## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

_(tu respuesta)_

**¿Qué patrón elegiste para tu proyecto, y qué número de latencia lo decide?**

**Por lotes.** El número que lo decide es la espera: la predicción alimenta el pedido de reposición, que se arma una vez al día, así que puede esperar **horas** (unas 6 horas entre que corre el lote a las 00:00 y se arman los pedidos a las 06:00, y hasta 24 horas sin que cambie la decisión), no milisegundos. Además los casos se conocen de antemano (10 tiendas × 3,049 productos = 30,490 series) y todas las features se conocen la noche anterior. Medí que el bosque final estima las 30,490 series de mañana en 0.16 s y los próximos 28 días (853,720 filas) en 4.5 s, mientras que una sola fila en línea tarda 25.7 ms. El detalle está en [`psets/proyecto/docs/despliegue.md`](../../psets/proyecto/docs/despliegue.md), y las 7 pruebas en [`psets/proyecto/modelado/test_pipeline_proyecto.py`](../../psets/proyecto/modelado/test_pipeline_proyecto.py), sobre [`psets/proyecto/modelado/pipeline_proyecto.py`](../../psets/proyecto/modelado/pipeline_proyecto.py), mi pipeline reorganizado en funciones para que se pueda importar sin leer los datos.

**Rompe a propósito una de tus pruebas (cambia un valor esperado), haz push y mira qué pasa en GitHub. Después arréglala. ¿Qué te dijo GitHub, y dónde lo viste?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Tendría que existir un **servicio en línea**: un programa siempre encendido (por ejemplo, una API con FastAPI) que tenga el modelo cargado en memoria y una ruta de predicción a la que la app le pueda hacer una petición. La app le mandaría los datos del viaje tal como los conoce en ese momento: zona de origen, zona de destino, distancia, número de pasajeros, método de pago y la fecha y hora del viaje. El servicio calcularía las features que faltan (`hora_sin`, `hora_cos`, `es_fin_de_semana`, `es_hora_pico`, `es_tarifa_fija`) con el mismo código del entrenamiento, y devolvería la tarifa estimada y la versión del modelo en milisegundos, antes de que el pasajero confirme el viaje.
