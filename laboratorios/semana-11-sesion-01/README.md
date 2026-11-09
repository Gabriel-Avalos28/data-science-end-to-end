# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

En un proyecto de Machine Learning, el autor destaca que se deben versionar tres componentes principales utilizando las siguientes herramientas:

| Componente | Herramienta |
| --- | --- |
| Código (Code) | Git |
| Datos (Data) | DVC (Data Version Control) |
| Modelo de Machine Learning (ML Model) | MLflow |

**¿Por qué no alcanza con Git para versionar los datos?**

Git no es suficiente para gestionar datos en proyectos de Machine Learning por las siguientes razones:

- **Manejo de archivos grandes y binarios:** Git está diseñado específicamente para rastrear cambios en archivos de texto plano (código fuente). No está optimizado para almacenar ni gestionar conjuntos de datos (datasets) masivos o archivos binarios pesados.
- **Sobrecarga del repositorio (Repository Bloat):** Intentar versionar grandes volúmenes de datos directamente en Git haría que el repositorio crezca de tamaño de forma desproporcionada.
- **Problemas de rendimiento:** Operaciones comunes como clonar el repositorio (git clone), subir cambios (git push) o descargar actualizaciones se volverían extremadamente lentas e imprácticas.

Por esta razón, herramientas como DVC trabajan sobre Git: almacenan los datasets reales en un almacenamiento externo o remoto, mientras que en Git solo guardan pequeños archivos de metadatos (.dvc) que actúan como punteros ligeros hacia la versión exacta de los datos.

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Tendría que saber todo lo que puede cambiar el número en cada corrida:

- **La división de los datos:** la semilla y el tamaño de prueba. Si cada uno mide sobre viajes distintos, los errores no son comparables.
- **La versión de los datos:** una huella del archivo, para confirmar que de verdad son los mismos, y el número de filas de entrenamiento.
- **Los hiperparámetros y la semilla del modelo:** por ejemplo, `n_estimators`, `learning_rate` y `random_state`.
- **La preparación y el código:** cómo se imputó, escaló y codificó, idealmente con el commit exacto.
- **La métrica:** que los dos midan lo mismo (por ejemplo, MAE) y sobre qué conjunto.
- **El ambiente:** las versiones de las librerías, como scikit-learn.

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Ninguno tiene "más razón": el de la división 42 da 1.2754 y el de la división 7 da 1.1867, pero cada uno se midió sobre un conjunto de prueba distinto, así que los números no son comparables. La salida permite ver que la división cambió (`division 42` y `division 7`). No dice si los datos fueron los mismos ni si los hiperparámetros coincidían, y tampoco queda guardada: desaparece al reiniciar el kernel.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La división de los datos. Espero ver distinto `semilla_division` (42 y 7), y que `huella_datos` y los hiperparámetros sean iguales. La tabla lo confirma: misma huella (`4800204d50c0`), mismo `n_estimators` (100) y solo cambia la semilla.

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Para poder usarlo sin volver a entrenar: cargarlo después, comprobar que reproduce su métrica y servirlo. Como se guarda el pipeline completo, incluye también la preparación ajustada (imputación, escalado y codificación). Los parámetros dicen cómo se entrenó, pero el artefacto es el resultado exacto de ese entrenamiento.

**Lo que mostró la tabla del paso 5:**

| Corrida | Error en prueba (MAE) | Semilla de la división | Huella de los datos |
| --- | --- | --- | --- |
| gradient boosting (compañero) | 1.1867 | 7 | `4800204d50c0` |
| gradient boosting | 1.2754 | 42 | `4800204d50c0` |
| bosque aleatorio | 1.3016 | 42 | `4800204d50c0` |

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

La red neuronal, con MAE en prueba de 1.1695. Le gana al gradient boosting (1.2754) por 0.1058, un 8.3% menos de error. Hay que tomar esa ventaja con cuidado: en el Ejercicio 1, el mismo gradient boosting pasó de 1.2754 a 1.1867 (0.0886 de diferencia) solo por cambiar la división, una variación casi tan grande como la ventaja de la red.

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Garantiza que el artefacto guardado es el pipeline que produjo esa métrica y que, cargado desde el registro, predice igual sobre los mismos datos de prueba (1.1695052296355013 en ambos casos). No garantiza que sea realmente el mejor ni que funcione igual con datos nuevos: el error viene de una sola división de 595 viajes, y además usamos ese mismo conjunto de prueba para elegirlo, así que el número puede ser optimista.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Las corridas nuevas tendrían un `huella_datos` distinto de `4800204d50c0`, porque cambiar un solo byte del archivo cambia el hash. Las métricas también podrían moverse, pero por sí solas no dirían por qué; la huella sí indica que los datos ya no son los mismos.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

Guarda todos sus hiperparámetros, también los que no escribimos y quedaron por defecto (`activation`, `solver`, `learning_rate_init`, etc.), junto con la huella de los datos, la semilla y las filas de entrenamiento. También muestra el tiempo de entrenamiento (4.34 s, el más lento de los cinco), la distancia entre el error de entrenamiento y el de prueba (1.0882 y 1.1695), cuándo y desde dónde se corrió, y conserva el modelo listo para cargarlo.

## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

### Entregable 1: `experimentos_proyecto.ipynb`

**Mi entrega:** [`proyecto/experimentos_proyecto.ipynb`](./proyecto/experimentos_proyecto.ipynb), con la copia de mi [`pipeline_proyecto.py`](./proyecto/pipeline_proyecto.py) al lado.

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

**Mi entrega:** [`proyecto/corridas_proyecto.csv`](./proyecto/corridas_proyecto.csv).

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

Elegí el **bosque aleatorio**, `run_id` `853a6190c5b64f2cb5aa9333c55445c0`.

| Modelo | MAE entrenamiento | MAE prueba | RMSE prueba | Segundos |
| --- | --- | --- | --- | --- |
| gradient boosting (absoluto) | 1.4370 | 1.4103 | 3.8576 | 3.1 |
| línea base (mediana) | 1.4370 | 1.4103 | 3.8576 | 0.6 |
| **bosque aleatorio (final)** | 1.5368 | 1.5069 | 3.1999 | 76.5 |
| red neuronal | 1.7012 | 1.6706 | 3.4048 | 195.1 |
| ridge alpha=10 | 1.8098 | 1.7202 | 3.5216 | 0.7 |

En MAE, mi métrica, el bosque no es el mejor: queda 0.0966 unidades (6.9%) por encima de la línea base. Pero la línea base y el gradient boosting absoluto predicen 0 unidades en todas las filas, porque el 59.8% de los días de entrenamiento no tienen ventas: ganan en MAE sin distinguir ningún producto, tienda o día. Entre los modelos que sí predicen ventas, el bosque tiene el menor MAE, y también tiene el menor RMSE de los cinco (3.1999, 17% menos que la línea base).

El notebook compara además el error según el tipo de día. En los 18,983 registros con ventas, el bosque se equivoca 2.146 unidades y "siempre 0" se equivoca 3.257, un 34% menos; el costo es equivocarse 1.019 unidades en los días sin ventas, donde "siempre 0" acierta por definición. En total predice 60,441 unidades de las 61,835 reales. No se sobreajusta: su error en prueba (1.5069) es incluso menor que en entrenamiento (1.5368). Al cargarlo desde el registro reproduce exactamente su MAE (1.5069454563287006).

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesita el mismo código (`pipeline_proyecto.py` y `experimentos_proyecto.ipynb`, en el mismo commit), exactamente el mismo archivo `muestra_con_features.csv.gz`, las mismas versiones de las librerías (`requirements.txt` o el Dev Container) y correr todo con las mismas semillas y la misma fecha de corte.

- **Registrado en MLflow:** los hiperparámetros con la semilla del modelo, la fecha de corte (`2016-01-01`), las filas de entrenamiento, la huella de los datos (`862b2ea566e9`), las métricas, el pipeline entrenado y las dependencias que MLflow guarda junto al modelo.
- **No registrado en MLflow:** el archivo de datos, porque solo está su huella: con ella se comprueba si es el mismo archivo, pero no se puede recuperar, y además `*.csv.gz` está en `.gitignore`. Tampoco quedó el commit del código. Y el registro completo vive en `.mlflow/`, que no se sube: quien clone el repositorio solo recibe `corridas_proyecto.csv` y tendría que conseguir los datos aparte.

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Cambiaría bastante. Repetí los dos mejores modelos de taxis con 10 divisiones distintas, en [`starter/variacion_divisiones.ipynb`](./starter/variacion_divisiones.ipynb). No lo registré en MLflow para no mezclar divisiones distintas en el experimento de comparación:

| Modelo | Promedio | Desviación estándar | Mínimo | Máximo |
| --- | --- | --- | --- | --- |
| red neuronal | 1.1898 | 0.0456 | 1.0954 | 1.2477 |
| gradient boosting | 1.2121 | 0.0527 | 1.0944 | 1.2851 |

La red ganó en 6 de las 10 divisiones, y en promedio solo por 0.0223. Es mucho menos que los 0.1058 de ventaja que mostró la división 42, así que esa ventaja dependía en buena parte de la división que tocó. En el registro casi no se ve: cada corrida guarda un solo número de una sola división. La única pista es el gradient boosting del Ejercicio 1, que pasó de 1.2754 a 1.1867 al cambiar la semilla de 42 a 7. Para medir esa variación habría que registrar varias divisiones o usar validación cruzada.
