# Semana 13, Sesión 1: Despliegue en producción: versionado y registro de modelos

> Objetivo de la sesión: versionar y registrar el modelo desplegado de tu proyecto, explicando qué problema resuelve un registro de modelos frente a sobrescribir el mismo archivo cada vez que se reentrena.

**Lectura previa de esta sesión:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 10: Infrastructure and Tooling for MLOps.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El registro de modelos es parte del mismo servidor de MLflow que usas desde la Semana 11: se levanta solo al iniciar el Codespace, en el puerto 5050 (si no responde, `bash .devcontainer/mlflow-server.sh`). Los scripts lo encuentran por la variable `MLFLOW_TRACKING_URI`. Los laboratorios de hoy son scripts.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la gobernanza de modelos, y qué componentes la forman?**

El artículo la define como "el proceso de punta a punta con el que una organización establece, implementa y mantiene controles sobre el uso de sus modelos", desde la documentación y el control de versiones hasta el back-testing, el monitoreo y la observabilidad. Sus componentes principales son:

- **Desarrollo y documentación del modelo:** el objetivo, la selección y validación de los datos de entrenamiento, y la metodología, los supuestos, el dataset y las salidas esperadas, por escrito.
- **Inventario de modelos:** un inventario central con todos los modelos en uso.
- **Validación y back-testing:** equipos independientes prueban el modelo con datos históricos y revisan sesgo, robustez y sobreajuste.
- **Despliegue y monitoreo:** vigilar de forma continua la deriva y la degradación, con controles de gobernanza dentro del pipeline de CI/CD.
- **Roles, responsabilidades y flujos de trabajo:** quién responde por el modelo (científicos de datos, riesgo, cumplimiento, auditores) en cada etapa, del desarrollo a la validación y al retiro.

**El artículo pide un inventario central de modelos. ¿Qué datos tendría que tener ese inventario sobre cada modelo, y para qué los necesita alguien que no lo entrenó?**

Según el artículo, el inventario registra de cada modelo **su propósito, su dueño, su metodología y su estado en el ciclo de vida** (en desarrollo, validado, en producción, retirado). Y debe incluir todos los modelos, no solo los de machine learning: también los de crédito y hasta los que viven en una hoja de cálculo. Para que le sirva a alguien que no lo entrenó, yo agregaría: qué **versión** está en producción, con qué **datos** se entrenó (su huella y la fecha de corte), con qué **métricas** se aprobó, y **dónde** está desplegado.

Quien no entrenó el modelo lo necesita para entender qué hace y para qué no debe usarse, para saber a quién preguntar, para evaluar su riesgo y decidir si se puede seguir usando (el artículo dice que el inventario "apoya una mejor evaluación de riesgos" y la toma de decisiones sobre su uso), y para responder a un auditor o regulador sin depender de la memoria del autor. La regulación bancaria SR 11-7 que cita el artículo exige justamente "un inventario completo de modelos". El registro de MLflow de hoy cumple ese papel para versión, corrida, métricas y alias.

## Pregunta de apertura

La sesión pasada, tu servicio cargaba `modelo.joblib`. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Con lo de la sesión pasada lo sabría solo en parte. Cada respuesta llevaba `version_modelo`, la huella MD5 de `modelo.joblib`, y `/salud` decía el `run_id`. Pero al reexportar, el archivo y `modelo_info.json` se reemplazan, y la huella no quedaba guardada en MLflow: para pasar de una huella vieja al modelo que la produjo no había una tabla fija. Volver al modelo anterior significaba cambiar a mano el `RUN_ID` en `exportar_modelo.py`, volver a exportar y reconstruir la imagen. Por suerte, MLflow guarda todos los runs, así que el modelo anterior no se pierde. Lo que faltaba es lo de hoy: un **registro de modelos** con versiones numeradas que no se pisan, y un alias (`produccion`) que se mueve de una versión a otra. Así, cada predicción dice qué versión la calculó, y volver atrás es mover el alias.

## Ejercicio 1: la tarifa, con registro

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa de la sesión pasada, ahora con el modelo en el registro de MLflow en vez de un archivo. [`registrar_modelo.py`](./starter/ejercicio1/registrar_modelo.py) entrena la red neuronal con el número de neuronas que le pases y la registra como una versión nueva del modelo `tarifa-taxis`. [`promover.py`](./starter/ejercicio1/promover.py) pone el alias `produccion` en la versión que le pases. [`versiones.py`](./starter/ejercicio1/versiones.py) lista las versiones con su error y sus alias (no lo modifiques). [`servicio.py`](./starter/ejercicio1/servicio.py) carga la versión que tiene el alias `produccion`. Los 4 `TODO`: registrar, promover, cargar desde el registro y responder qué versión calculó cada tarifa.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python registrar_modelo.py 64
python versiones.py
python promover.py 1
pytest -v
```

**Respuesta 1 (antes de los TODO):** `registrar_modelo.py` dice `Version None`, `versiones.py` no imprime nada y `promover.py` termina con un error. ¿Qué dice el error, y qué te dice sobre lo que hay en el registro?

El error final de `promover.py` es:

```
mlflow.exceptions.RestException: INVALID_PARAMETER_VALUE: Registered model alias produccion not found.
```

Dice que en el registro no hay nada con el alias `produccion`: no hay ninguna versión a la que se le haya puesto, ni siquiera una versión registrada. `registrar_modelo.py` sí entrenó y guardó la corrida con su MAE (1.1695), pero nunca guardó el modelo ni lo registró. Por eso dice `Version None` y `versiones.py` no tiene nada que listar. La corrida está en el experimento; el modelo no está en el registro.

**Respuesta 2 (TODO 1):** registra dos versiones, una con 64 neuronas y otra con 8. Antes de correr `versiones.py`, ¿cuántas versiones esperas ver, y qué pasó con la primera cuando registraste la segunda?

Esperaba **dos versiones**, y la primera **intacta**. Registrar la segunda no reemplaza a la primera: el registro le da un número nuevo y conserva las dos, cada una con su corrida y su métrica. Fue lo que salió:

```
Version 1: red de 64 neuronas, MAE 1.1695, alias: -
Version 2: red de 8 neuronas, MAE 1.1877, alias: -
```

En la Semana 12, en cambio, reentrenar reemplazaba `modelo_tarifa.joblib`, y el modelo anterior se perdía. Hubo un error en clase: `registrar_modelo.py` cargaba `mlflow.sklearn.load_model("MODELO")` para sacar la versión. Eso fallaba con `No such file or directory: 'MODELO'`, y además un Pipeline no tiene versión. La versión ya la devuelve `log_model`: `info.registered_model_version`.

**Respuesta 3 (TODO 2 y 3):** pon `produccion` en la versión 2 y levanta el servicio (`uvicorn servicio:app --port 8000`). ¿Qué versión dice `/salud`? Ahora mueve el alias a la versión 1 sin reiniciar uvicorn: ¿qué versión responde el servicio? ¿Y después de reiniciarlo?

Con `produccion` en la versión 2, `/salud` dijo `{"estado":"ok","modelo_cargado":true,"version_modelo":"2"}`. Moví el alias a la versión 1 con `python promover.py 1` **sin reiniciar** uvicorn, y el servicio **siguió respondiendo la versión 2**, con la misma tarifa. El servicio lee el alias y carga el modelo **una sola vez, al arrancar**; mover el alias cambia el registro, no lo que ya está en memoria. **Después de reiniciarlo**, `/salud` dijo `"version_modelo":"1"`. Me pasó también por accidente: mi uvicorn del puerto 8000 seguía diciendo versión `"3"` de un registro que ya había borrado. En clase hubo otro error: la URI de carga estaba sin `f` (`'models:/{NOMBRE}/{version_servida}'`), y MLflow buscaba un modelo llamado literalmente `{NOMBRE}`.

**Respuesta 4 (TODO 4):** las pruebas usan un modelo de prueba y una versión inventada, `"7"`, sin servidor de MLflow. ¿Qué prueba falla antes del `TODO 4`, y qué comprueba que el `/salud` no comprobaba?

Falla `test_cada_prediccion_dice_que_version_la_calculo`: la respuesta era `{'tarifa_estimada': 20.0}` y la prueba espera `{'tarifa_estimada': 20.0, 'version_modelo': '7'}`. `/salud` dice qué versión tiene cargada el servicio **ahora**. La prueba nueva comprueba que **cada tarifa** lleva la versión que la calculó, que es lo que queda guardado en la app junto a la tarifa. Si mañana un pasajero reclama, `/salud` ya puede estar sirviendo otra versión, pero la respuesta que recibió dice cuál fue. Ahora pasan las dos: `2 passed`.

**Lo que respondió el servicio:**

Viaje de Midtown al aeropuerto JFK (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta):

| Alias `produccion` en | Respuesta de `/predecir` |
|---|---|
| Versión 2 (8 neuronas) | `{"tarifa_estimada": 69.98, "version_modelo": "2"}` |
| Versión 1 (64 neuronas), sin reiniciar | `{"tarifa_estimada": 69.98, "version_modelo": "2"}` (sigue la 2) |
| Versión 1 (64 neuronas), después de reiniciar | `{"tarifa_estimada": 70.07, "version_modelo": "1"}` |

El alias quedó en la versión 1, la de menor MAE (1.1695 contra 1.1877).

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el ciclo de vida del modelo de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El equipo de fraude entrena modelos candidatos cada tanto, y quiere una regla: una versión llega a `produccion` solo si detecta más fraudes (mayor recall al umbral de 0.10) que la que está ahí. [`ciclo_fraude.py`](./starter/ejercicio2/ciclo_fraude.py) entrena tres candidatos, registra cada uno y decide si lo promueve. Los pasos 1 a 3 tienen `TODO`: registrar una versión con sus métricas, leer la métrica de una versión y averiguar qué versión está en producción. Los pasos 4 (decidir y mover el alias) y 5 (etiquetar la versión con el resultado) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python ciclo_fraude.py`.

**Respuesta 1:** ¿qué versión quedó en producción, y por qué no la última que se registró?

Quedó en producción la **versión 2, el gradient boosting**:

```
logistica: version 1, recall 0.644, precision 0.304, promovida: True
gradient-boosting: version 2, recall 0.661, precision 0.429, promovida: True
logistica-muy-regularizada: version 3, recall 0.102, precision 0.667, promovida: False
En produccion: version 2
```

La última registrada fue la 3, pero la regla no promueve a la más nueva: promueve a la que detecta más fraudes que la que está en producción. La 3 tiene un recall de 0.102 contra 0.661 de la 2, así que se rechazó. `comprobar.py` da los cinco pasos en `OK`.

**Respuesta 2:** la regresión logística muy regularizada tiene la mayor precisión de las tres. ¿Por qué la regla la rechazó? ¿Qué tendría que cambiar en la regla si la empresa decide que las falsas alarmas cuestan más que los fraudes no detectados?

Porque la regla solo mira el **recall**. Esa versión casi no marca pagos: de los que marca, el 66.7% son fraude (la mayor precisión), pero solo detecta el **10%** de los fraudes, contra el 66% del gradient boosting. Con el costo de la Semana 11 (un fraude no detectado cuesta 40 dólares y una falsa alarma 3), dejar pasar el 90% de los fraudes es lo más caro. Si las falsas alarmas costaran más que los fraudes no detectados, la regla tendría que comparar la **precisión**, o mejor, el **costo total** con los costos nuevos (fraudes no detectados por su costo, más falsas alarmas por el suyo). Probablemente también habría que revisar el umbral de 0.10, que se eligió para costos al revés.

**Respuesta 3:** abre el modelo `fraude-pagos` en la interfaz de MLflow (puerto 5050, pestaña **Models**). Si dentro de un mes alguien pregunta por qué la versión 3 nunca llegó a producción, ¿dónde está la respuesta?

En **la propia versión 3**, en el registro. En la pestaña **Models** → `fraude-pagos` → **Version 3** están sus etiquetas: `validacion = rechazada` y `comparada_con = 2`, que dicen que se comparó con la versión 2 y no la superó. Desde ahí se abre la corrida que la produjo (`logistica-muy-regularizada`), con su recall de 0.102 y su precisión de 0.667, y la corrida de la versión 2 muestra el recall de 0.661 con que se comparó. La decisión queda escrita junto al modelo, no en la memoria de alguien ni en un chat.

**Si terminas antes:** vuelve a correr `python ciclo_fraude.py`. ¿Cuántas versiones hay ahora, y cuál está en producción? ¿Es lo que esperabas?

Ahora hay **6 versiones**, y en producción **sigue la versión 2**:

```
logistica: version 4, recall 0.644, precision 0.304, promovida: False
gradient-boosting: version 5, recall 0.661, precision 0.429, promovida: False
logistica-muy-regularizada: version 6, recall 0.102, precision 0.667, promovida: False
En produccion: version 2
```

Sí, es lo esperado. Cada corrida registra versiones nuevas, aunque sean idénticas: mismos datos, misma semilla, mismas métricas. Ninguna supera a la 2: la 5 es el mismo gradient boosting con el mismo recall de 0.661, y la regla pide un recall *estrictamente* mayor, así que no se mueve el alias por un empate. Eso es bueno: reentrenar lo mismo no cambia producción. Pero el registro se llena de duplicados, etiquetados `rechazada` con `comparada_con = 2`.

## Ejercicio 3: el registro de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio que construiste la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: el modelo, en el registro

- `servicio/registrar_modelo.py`: entrena tu modelo final, guarda su métrica principal en la corrida y lo registra como una versión nueva de un modelo con el nombre de tu proyecto. Cada vez que lo corres crea una versión; ninguna reemplaza a otra.
- `servicio/promover.py`: pone el alias `produccion` en la versión que le pases.
- Al menos dos versiones registradas (por ejemplo, con dos valores de un hiperparámetro), y una de ellas con el alias `produccion`.

Corre los scripts desde `psets/proyecto/` como módulos (`python -m servicio.registrar_modelo`), para que encuentren `modelado/`.

### Entregable 2: el servicio sabe qué versión sirve

- El script que exportaba el modelo (Semana 12) ahora exporta la versión con el alias `produccion`, y guarda su número junto al modelo (por ejemplo, en `servicio/modelo_version.txt`). La imagen lleva los dos archivos adentro: el contenedor no necesita llegar al servidor de MLflow.
- `/salud` y cada predicción responden `version_modelo`.
- Una prueba en `servicio/test_api.py` que compruebe que la predicción incluye la versión, con un modelo de prueba y una versión inventada.

### Entregable 3: `docs/despliegue.md`

Agrega la sección **Versionado y registro**: cómo se llama tu modelo en el registro, qué versión está en producción y con qué métrica, con qué criterio se mueve el alias, y los pasos para volver a la versión anterior.

**¿Con qué criterio decides que una versión nueva de tu modelo pasa a producción? ¿Quién lo decide en tu proyecto, y dónde queda registrada la decisión?**

Una versión nueva de `pronostico-ventas-tiendas` pasa a producción solo si, entrenada con los mismos datos y la misma fecha de corte (la misma huella, `862b2ea566e9`), tiene un **costo de stock de prueba menor** que la versión en producción. Ese costo cuenta una unidad que falta como 3 veces una que sobra (Semana 11). El MAE se mira como control. Hoy registré dos versiones: la 1 (`max_depth=12`, el bosque final de la Semana 11) con un costo de 3.0457 y un MAE de 1.5069, y la 2 (`max_depth=6`) con 3.1486 y 1.5709. Por eso quedó la **versión 1** en `produccion`.

Lo decido yo, como responsable del modelo, con `python -m servicio.promover <version>`, que lista las métricas de todas las versiones antes de mover el alias. La decisión queda registrada en tres lugares: el **alias** `produccion` en el registro de MLflow, la **etiqueta** `promovida_con` en la versión promovida, y la sección **Versionado y registro** de [`docs/despliegue.md`](../../psets/proyecto/docs/despliegue.md#versionado-y-registro), con la tabla de versiones y los pasos para volver a la anterior. Los scripts están en [`psets/proyecto/servicio/`](../../psets/proyecto/servicio/): `registrar_modelo.py`, `promover.py` y `exportar_modelo.py`.

**Si tu servicio corre en un contenedor sin acceso al servidor de MLflow, ¿cómo sabe qué versión está sirviendo?**

Porque el número viaja **dentro de la imagen**, junto al modelo. `exportar_modelo.py` busca en el registro la versión con el alias `produccion`, guarda su modelo en `servicio/modelo.joblib` y su número en `servicio/modelo_version.txt`. El `Dockerfile` copia los dos archivos. Al arrancar, `api.py` lee el número una vez, y `/salud` y cada respuesta de `/predecir` lo devuelven:

```
$ curl localhost:8000/salud
{"estado":"ok","modelo_cargado":true,"version_modelo":"1","series_con_referencia":30462}
$ curl -X POST localhost:8000/predecir -H "Content-Type: application/json" \
    -d '{"fecha": "2016-05-09", "item_id": "FOODS_3_090", "store_id": "TX_2", "precio": 1.5}'
{"unidades_estimadas":7.45,"serie_conocida":true,"version_modelo":"1"}
```

La contracara: si alguien mueve el alias en MLflow pero no vuelve a exportar ni a construir la imagen, el contenedor sigue sirviendo la versión anterior, aunque responde la verdad sobre cuál es. Lo probé: con el alias en la versión 2 y exportando otra vez, el servicio respondió `"version_modelo": "2"` (3.50 unidades), y al volver a la 1, otra vez `"1"` (7.45). La prueba `test_salud_y_prediccion_dicen_que_version_sirven` lo comprueba con una versión inventada, `"7"`; `pytest psets/proyecto` da `19 passed`.

## Antes de la próxima sesión

**Lectura:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Se degrada **en silencio**. Sigue respondiendo en milisegundos y con código 200, `/salud` dice `ok`, y cada predicción trae su versión. Pero nada de eso mide si la predicción acierta. Si los datos cambian (otra mezcla de viajes, un aumento de tarifas, un feriado, productos nuevos en mi proyecto) o cambia la relación entre las features y el objetivo, el modelo sigue aplicando lo que aprendió y se equivoca cada vez más, sin ningún error que lo delate. Las métricas de sistema (latencia, errores, CPU) no lo detectan.

Para enterarme hay que **monitorear el modelo, no solo el servicio**:
- Guardar cada predicción con su entrada, su versión y la hora.
- Cuando llegue el valor real (la tarifa cobrada, las unidades vendidas al día siguiente), unirlo con la predicción y calcular el error por día y por versión. En mi proyecto, el costo de stock diario, comparado con el 3.05 de la prueba.
- Mientras el valor real no llega, vigilar la **deriva de los datos de entrada**: comparar la distribución de las features de hoy (precio, distancia, día de la semana) con la del entrenamiento, y la de las predicciones.
- Poner alertas con umbrales, para que me avise el sistema y no un usuario que reclama.
