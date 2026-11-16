# Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

Nada nuevo que instalar: FastAPI y uvicorn ya llegaron con el contenedor reconstruido la sesión pasada. Los laboratorios de hoy son scripts y un `Dockerfile`. El servicio escucha en el puerto 8000: cuando lo levantes, aparece en la pestaña **Ports** como "Servicio de predicción (Semana 12)", y su documentación interactiva queda en la ruta `/docs` de ese puerto.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué pasos separan un modelo entrenado en un notebook de un modelo que otra aplicación puede usar?**

1. **Guardar el modelo entrenado** en un archivo (serializarlo con pickle o joblib), junto con su preparación, para no volver a entrenarlo cada vez que se necesita una predicción.
2. **Envolverlo en una API web** (Flask o FastAPI) con una ruta que reciba los datos de entrada, aplique la misma preparación del entrenamiento, llame a `predict` y devuelva la predicción como JSON. Ese es el contrato que otra aplicación entiende, sin saber nada de Python ni del notebook.
3. **Empaquetarlo en un contenedor** (Docker) con sus dependencias en versiones fijas.
4. **Desplegarlo en un servidor o en la nube**, donde quede encendido y accesible para las otras aplicaciones.

Es lo que hicimos hoy: `entrenar.py` deja el `.joblib`, `servicio.py` es la API, el `Dockerfile` la empaqueta y `test_servicio.py` prueba el contrato.

**¿Para qué usa el artículo un contenedor, si el servicio ya corre en la máquina de quien lo escribió?**

Para que el servicio no dependa de la máquina de quien lo escribió. En su computador funciona porque ahí están instaladas las librerías correctas, en las versiones correctas, con el modelo en la carpeta correcta. En otro servidor eso no está garantizado, y un modelo guardado con joblib se carga bien solo con la misma versión de scikit-learn. El contenedor lleva todo junto (Python, las librerías fijadas en `requirements.txt`, el código y el modelo), así que corre igual en cualquier lado: en otra máquina, en la nube o en varias copias a la vez si hay mucho tráfico. También tiene un costo, como vimos en la Respuesta 5: dentro del contenedor hay que escuchar en `0.0.0.0`, porque su `127.0.0.1` es solo suyo.

## Pregunta de apertura

La sesión pasada, la app leía las tarifas de una tabla calculada de noche. Cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Tendría que existir un **servicio en línea**: un programa siempre encendido, por ejemplo una API con FastAPI, que tenga el modelo cargado en memoria y una ruta de predicción a la que la app le pueda hacer una petición. La app le mandaría los datos del viaje tal como los conoce en ese momento: zona de origen, zona de destino, distancia, número de pasajeros, método de pago y la fecha y hora del viaje. El servicio calcularía las features que faltan (`hora_sin`, `hora_cos`, `es_fin_de_semana`, `es_hora_pico`, `es_tarifa_fija`) con el mismo código del entrenamiento, y devolvería la tarifa estimada en milisegundos, antes de que el pasajero confirme el viaje.

## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

Que todo se ve bien y nada funciona. `/predecir` responde 200, pero siempre con una tarifa de 0.0, porque está escrita a mano y el modelo nunca se usa. `/salud` solo dice que el archivo se cargó, no que las predicciones tengan sentido. El esquema solo pedía las zonas, así que aceptaba un viaje sin fecha ni distancia y descartaba el resto en silencio. Y `3 passed` engaña: dos de las tres pruebas no tenían ningún `assert`. Un código 200 dice que el servicio respondió, no que respondió bien.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

Un viaje sin `distancia_km` recibe **200 con una tarifa**: el campo es opcional (`float | None = None`), llega como vacío y la preparación del pipeline lo imputa con la mediana del entrenamiento. El viaje de Midtown a JFK sin distancia recibió 68.56 (con 26.4 km recibe 70.39). Uno sin `zona_origen` recibe **422**, porque la zona es obligatoria en el esquema y el servicio lo rechaza antes de predecir. En clase habíamos declarado los opcionales como `float | None` sin `= None`; en Pydantic 2 eso los deja obligatorios aunque acepten `null`, así que les agregué el valor por defecto.

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

El modelo recibiría una feature con el mismo nombre pero con otro significado, y se equivocaría **sin dar ningún error** (training-serving skew). Por ejemplo, si el servicio marcara como hora pico las 10:00, a esos viajes les sumaría el efecto que aprendió para el tráfico de 7 a 9 y de 16 a 19, y a los viajes reales de hora pico se lo quitaría. Las pruebas de forma seguirían pasando y el JSON seguiría siendo válido. Por eso el servicio no calcula nada por su cuenta: importa `agregar_features` de `features_taxis.py`, el mismo módulo con que se construyeron las features del entrenamiento. En la versión de clase había otro error: a `agregar_features` se le pasaban solo las columnas del modelo, que todavía no existían y además dejaban fuera `fecha_hora_inicio`; y después del `predict` quedaba `tarifa = 0.0`, que pisaba la predicción. Lo corregí pasándole el viaje completo y tomando la primera predicción.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

Lo produce **FastAPI**, con Pydantic. Antes de llamar a `predecir`, FastAPI valida el JSON contra el esquema `Viaje`; como falta `zona_origen`, responde 422 con el detalle (`"type": "missing", "loc": ["body", "zona_origen"]`) y mi función nunca se ejecuta. La petición **no llega al modelo**. La prueba del `TODO 4` comprueba el código 422 y también que el error señala `zona_origen`.

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Porque sin `--host`, uvicorn escucha en `127.0.0.1`, que **dentro del contenedor** es el localhost del propio contenedor. El log lo dice: `Uvicorn running on http://127.0.0.1:8000`. `docker run -p 8000:8000` reenvía las peticiones a la interfaz de red del contenedor, no a su localhost, y ahí no escucha nadie: `curl` falla con la conexión cortada (código de salida 56) aunque el servicio esté vivo. Con `CMD ["uvicorn", "servicio:app", "--host", "0.0.0.0", "--port", "8000"]` escucha en todas las interfaces, el log dice `http://0.0.0.0:8000` y `/salud` responde `{"estado":"ok","modelo_cargado":true}`.

**Lo que respondió el servicio:**

Viaje de Midtown al aeropuerto JFK (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta): **`{"tarifa_estimada": 70.39}`**, lo mismo con `uvicorn` en el Codespace que con el contenedor.

El mismo viaje sin zona de origen, **HTTP 422**:

```
{"detail":[{"type":"missing","loc":["body","zona_origen"],"msg":"Field required","input":{"fecha_hora_inicio":"2026-11-05 08:30","zona_destino":"JFK Airport","distancia_km":26.4,"pasajeros":2,"metodo_pago":"tarjeta"}}]}
```

Las 3 pruebas pasan: `3 passed`.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

El de madrugada (02:40, app, tarjeta nueva, 2 intentos fallidos, 0 viajes previos, 41.30 dólares) recibió **`{"probabilidad_fraude": 0.9996, "revisar": true}`**. El de la tarde (14:10, tarjeta conocida, 0 intentos fallidos, 22 viajes previos, 19.80 dólares) recibió **`{"probabilidad_fraude": 0.0159, "revisar": false}`**. `comprobar.py` da los cinco pasos en `OK`, y `pytest` da `3 passed`.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app recibe una decisión lista y no tiene que saber cuál es el umbral: solo manda el pago a revisión o no. La regla del negocio vive en un solo lugar, y todas las apps que llaman al servicio (Android, iOS, web) deciden igual. Si la empresa cambia el umbral a 0.20, solo cambia `UMBRAL` en `servicio_fraude.py` (mejor si se lee de una variable de entorno, como `RUTA_MODELO`) y se reconstruye y vuelve a desplegar la imagen; la app no cambia. Hay que cambiarlo también en el lote de la sesión pasada, que tiene su propio `UMBRAL`, o los dos dejarían de decidir lo mismo. La prueba del pago válido seguiría pasando (0.25 también alcanza 0.20), así que el cambio de umbral no lo detecta ninguna prueba: es una decisión del costo, como en la Semana 11.

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Sí. Mandé al servicio el pago 70001 de `pagos_ayer.csv` y respondió `0.2628`, la misma probabilidad que le dio el lote de la sesión pasada en `pagos_puntuados.csv`. Para que así sea tiene que ser igual: **el mismo modelo** (los dos `modelo_fraude.joblib` tienen la misma huella, `377bc8b74cf7`, porque se entrenaron con los mismos 6000 pagos y la misma semilla), **la misma preparación** (`preparacion_pagos.py`, con las mismas `agregar_features` y `columnas`, en el mismo orden), **las mismas versiones de las librerías** (scikit-learn 1.5.2) y **los mismos datos de entrada** (la misma fecha y hora, que da la misma `hora`, y el mismo monto y distancia, que dan el mismo `monto_por_km`).

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde **HTTP 422**, sin llegar al modelo:

```
{"detail":[{"type":"float_parsing","loc":["body","monto"],"msg":"Input should be a valid number, unable to parse string as a number","input":"cuarenta"}]}
```

Lo produce la línea `monto: float` del esquema `Pago`. No hay un `if` en mi código: FastAPI valida el JSON con Pydantic antes de llamar a `evaluar_pago`, y como `"cuarenta"` no se puede convertir a número, rechaza la petición.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

Recibe lo que conoce quien pide la predicción del día: `fecha`, `item_id`, `store_id`, `precio` y `es_dia_de_evento` (si es feriado o evento del calendario, por defecto `false`). El servicio calcula todas las demás columnas del modelo con `preparar_peticiones`, importada desde `modelado/pipeline_proyecto.py`:

- `agregar_calendario`: `WEEKDAY` e `IS_WEEKEND` desde la fecha, `IS_SNAP` desde el estado de la tienda y el día del mes, y `CAT_ID` desde el `item_id` (`FOODS_3_090` → `FOODS`).
- `agregar_features`, la misma de siempre: `dias_desde_lanzamiento` y `precio_relativo`, con la fecha de lanzamiento y el precio promedio de entrenamiento de cada serie, que vienen de `servicio/referencias.csv`. También pasa las banderas a 1/0.

Sé que las calcula igual por cuatro cosas:
1. No están copiadas: el servicio y el entrenamiento usan la misma `agregar_features`.
2. Una prueba, `test_preparar_peticiones_da_las_mismas_features_que_el_entrenamiento`, compara las dos rutas columna por columna, y otra, `test_el_servicio_calcula_las_features_como_en_el_entrenamiento`, revisa con un modelo espía lo que llega al modelo (lunes, sin SNAP, 22 días, precio relativo 0.8).
3. Lo comprobé sobre las 468,820 filas reales de la muestra: el servicio, a partir de solo esas cinco columnas, da exactamente las mismas features que el entrenamiento.
4. Para la fila real de `FOODS_3_090` en `TX_2` el 2016-05-09, el contenedor responde 7.45 y el modelo leído directo de MLflow da 7.4538.

El servicio está en [`psets/proyecto/servicio/`](../../psets/proyecto/servicio/), con 9 pruebas en `test_api.py`; con las del pipeline, `pytest psets/proyecto` da `18 passed`.

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

Con el contenedor levantado (`docker build -f servicio/Dockerfile -t servicio-demanda .` y `docker run -p 8000:8000 servicio-demanda`, desde `psets/proyecto/`):

```
$ curl localhost:8000/salud
{"estado":"ok","modelo_cargado":true,"version_modelo":"6ba4baa93c74","run_id":"853a6190c5b64f2cb5aa9333c55445c0","series_con_referencia":30462}

$ curl -X POST localhost:8000/predecir -H "Content-Type: application/json" \
    -d '{"fecha": "2016-05-09", "item_id": "FOODS_3_090", "store_id": "TX_2", "precio": 1.5, "es_dia_de_evento": false}'
{"unidades_estimadas":7.45,"serie_conocida":true,"version_modelo":"6ba4baa93c74"}

$ curl -X POST localhost:8000/predecir -H "Content-Type: application/json" \
    -d '{"fecha": "2016-05-09", "item_id": "FOODS_3_090", "store_id": "NY_1", "precio": 1.5}'
{"detail":[{"type":"string_pattern_mismatch","loc":["body","store_id"],"msg":"String should match pattern '^(CA|TX|WI)_\\d+$'","input":"NY_1","ctx":{"pattern":"^(CA|TX|WI)_\\d+$"}}]}   (HTTP 422)
```

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Hoy lo sabría solo en parte. Cada respuesta de mi servicio lleva `version_modelo`, la huella del `modelo.joblib` que la produjo, y `/salud` dice de qué `run_id` de MLflow salió. Si quien reclama guardó la respuesta, o si el servicio guarda un log de cada petición con su respuesta (que todavía no guarda), sé qué archivo la produjo. Pero la huella no está registrada en MLflow, así que para pasar de la huella al run dependo de `modelo_info.json`, que se reemplaza en cada exportación. Para volver al modelo anterior, hoy tendría que cambiar el `RUN_ID` en `exportar_modelo.py`, volver a exportar y reconstruir la imagen; MLflow guarda todos los runs con su modelo, así que el modelo anterior no se pierde. Lo que falta es una relación fija entre versión, modelo y despliegue: registrar el modelo en el Model Registry de MLflow con versiones (y un alias como "produccion" que se mueve de una versión a otra), etiquetar cada imagen de Docker con esa versión y guardar las predicciones con su versión. Así, volver atrás sería mover el alias o levantar la imagen anterior.
