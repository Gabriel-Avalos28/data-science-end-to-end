# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada, y tu `pipeline_proyecto.py`.

Como la sesión pasada, los laboratorios son notebooks que importan la preparación de un script (`preparacion_taxis.py` en los Ejercicios 1 y 2, tu `pipeline_proyecto.py` en el 3).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

El aprendizaje automático clásico conviene para datos estructurados o tabulares, con una etiqueta bien definida y conjuntos de datos modestos, y cuando importa la interpretabilidad o la baja latencia; por ejemplo, mantenimiento predictivo o detección de fraude. Un LLM conviene cuando la entrada es texto no estructurado: generar texto, entender lenguaje, atención al cliente o análisis de documentos. El artículo aclara que no compiten: en la práctica muchas organizaciones combinan los dos.

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

El costo. Mis datos son una tabla estructurada, y un modelo clásico entrena en mi CPU en menos de 2 minutos y después predice casi gratis. Con un LLM tendría que pagar cada predicción todos los días: para 3049 productos en 10 tiendas calculé unos 224 dólares al mes solo en tokens de entrada, y no tengo evidencia de que predijera mejor.

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

- **Costo:** de entrenar y, sobre todo, de cada predicción, multiplicado por los viajes del mes.
- **Latencia:** cuánto tarda en responder, porque el pasajero está esperando la tarifa en la app.
- **Consistencia:** si el mismo viaje siempre recibe la misma tarifa.
- **Explicabilidad:** si puedo decirle al conductor o al pasajero por qué salió ese número.
- **Peores errores:** no solo el promedio, sino cuánto falla en casos raros como el aeropuerto.

## Ejercicio 1: una red neuronal frente al ensamblaje

Notebook: [`starter/ejercicio1_red_neuronal.ipynb`](./starter/ejercicio1_red_neuronal.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). Ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división, importadas de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

Significa que la red llegó al límite de iteraciones antes de terminar de aprender: sus pesos todavía estaban cambiando. Con hasta 1000 iteraciones esperaba que convergiera y que bajara su error. Así fue: la red no llegó al límite y su error en prueba quedó en 1.1695, mejor que el 1.2754 del gradient boosting.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

1729. De la entrada a la capa oculta son 25 × 64 = 1600 pesos más 64 interceptos, y de la capa oculta a la salida son 64 pesos más 1 intercepto: 1600 + 64 + 64 + 1 = 1729. La salida del notebook da el mismo número.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

Esperaba la distancia, y salió `distancia_km` con 0.9824 de importancia. El gradient boosting permite responder mejor, porque sus importancias muestran qué columnas pesan más y una tarifa de 70 se explica por la distancia o por `es_tarifa_fija`, aunque esas importancias son del modelo completo y no de un viaje en particular. En la red son 1729 números sin significado directo, así que no se le puede dar una razón clara al conductor.

**Lo que confirmó la salida, TODO por TODO:**

| Modelo | MAE entrenamiento | MAE prueba | Segundos |
| --- | --- | --- | --- |
| gradient boosting | 1.0705 | 1.2754 | 0.26 |
| red neuronal | 1.0882 | 1.1695 | 4.35 |

La red tiene 1729 parámetros. Las 3 columnas más importantes para el ensamblaje son `distancia_km` (0.9824), `es_tarifa_fija` (0.0120) y `es_hora_pico` (0.0028).

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Notebook: [`starter/ejercicio2_tres_soluciones.ipynb`](./starter/ejercicio2_tres_soluciones.ipynb), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre las celdas hasta la que escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y corre sus celdas y la de la comprobación hasta que todas digan `OK`. Los modelos ya están entrenados en el kernel: no hace falta volver a correr las celdas de arriba. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

Usé tres: Claude (Opus 5.5), Gemini (Gemini 4 Argon) y ChatGPT (GPT-6). El script usa la respuesta de Claude (`respuestas_llm.txt`); las otras dos están en `respuestas_gemini.txt` y `respuestas_chatgpt.txt`. El prompt tiene 2902 caracteres, unos 726 tokens, y 10 000 viajes son 500 prompts: 363 mil tokens de entrada al mes.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.8957 | 0.29 s en mi CPU | Casi 0: predice en milisegundos en el mismo servidor | Sí: `feature_importances_` dice qué columnas pesan más (la distancia) |
| Red neuronal | 0.9637 | 3.86 s en mi CPU | Casi 0, igual que el boosting | Difícil: son miles de pesos sin significado directo |
| LLM: Claude Opus 5.5 | 1.1375 | 0 (no se entrena, solo 30 ejemplos en el prompt) | 1.45 dólares ($4 por millón de tokens de entrada) | Te da una explicación en palabras, pero no hay garantía de que sea la razón real del número |
| LLM: Gemini 4 Argon | 1.2685 | 0 | 0.73 dólares con la tarifa de lanzamiento ($2), 1.45 desde diciembre ($4) | Igual que Claude |
| LLM: ChatGPT (GPT-6) | 0.9035 | 0 | Entre 0.04 y 0.73 dólares según la variante ($0.10 a $2); ChatGPT no me dijo cuál usó | Igual que Claude |

Ojo: el costo ignora los tokens de salida, como pide el ejercicio, pero en los modelos con razonamiento esos tokens se cobran 5 veces más caro y pueden ser más que los de entrada (Gemini estimó unos 2100 de salida por prompt). El costo real sería varias veces mayor.

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

No. Corrí el mismo prompt en una conversación nueva de Claude ([`starter/respuestas_claude_2.txt`](./starter/respuestas_claude_2.txt)) y 17 de los 20 números cambiaron, con una diferencia promedio de 0.39 dólares y de hasta 1.30 (viaje 10: 21.30 y 20.00). Solo se repitieron los 3 viajes de tarifa fija al aeropuerto (70). Incluso el error cambió, de 1.1375 a 1.0165, sin tocar nada. Entre los tres LLM pasó lo mismo: con el mismo prompt dieron números distintos en 17 de los 20 viajes, con diferencias de hasta 2.30 dólares (viaje 12, Upper West Side a Midtown: 12.70, 12.23 y 10.40). Solo coincidieron en los 3 viajes de tarifa fija al aeropuerto (70). Para una app que cobra, eso es un problema: el mismo viaje no puede costar distinto según cuándo se pregunte, es injusto con el pasajero y no se puede auditar. El gradient boosting siempre da el mismo número para los mismos datos.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

La peor de los tres LLM fue el mismo viaje: de JFK Airport a JFK Airport, 1.01 km, con tarifa real de 11.46. Claude dijo 7.70, Gemini 6.53 y ChatGPT 7.50, errores de 3.8 a 4.9 dólares. Lo particular es que es un viaje corto dentro del aeropuerto: no es tarifa fija (`es_tarifa_fija = 0`), pero cuesta más que un viaje de 1 km en la ciudad (en los ejemplos, 0.94 km en Midtown costó 7.46). Los LLM extrapolaron con la regla "1 km ≈ 7 dólares" porque en los 30 ejemplos no hay ningún viaje corto en JFK. El ensamblaje sí lo vio en las 2380 filas de entrenamiento y predijo 12.74, a 1.28 del real. Con solo 30 ejemplos el LLM no puede aprender los casos raros.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

No se mide sobre los 595 porque habría que pegar los 595 viajes en el chat (30 prompts de 20 viajes, o uno muy largo), copiar a mano 595 números y pagar 30 veces más; además, un prompt muy largo aumenta el riesgo de que el LLM se salte o desordene filas. Pero una comparación sobre 20 es poco confiable: un solo viaje cambia mucho el promedio. Se ve en mis números: en los 20 viajes el boosting (0.90) le gana a la red (0.96), pero en los 595 es al revés (1.28 contra 1.17). Y sin el viaje de JFK a JFK, Claude bajaría de 1.14 a cerca de 1.0. Con 20 viajes solo puedo decir que los tres quedan parecidos, no cuál es mejor.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.ipynb`

**Mi entrega:** [`proyecto/comparar_proyecto.ipynb`](./proyecto/comparar_proyecto.ipynb), con la tabla en [`proyecto/comparar_proyecto.csv`](./proyecto/comparar_proyecto.csv) y la copia de mi [`pipeline_proyecto.py`](./proyecto/pipeline_proyecto.py) al lado. El bosque aleatorio dio MAE 1.5069 en 80.6 segundos, y la red neuronal 1.6706 en 216.9 segundos, con 1793 parámetros.

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí tu preparación y tu división, y entrene el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

El LLM fue Claude Opus 5.5. El prompt está en [`proyecto/prompt_proyecto.txt`](./proyecto/prompt_proyecto.txt) y sus respuestas en [`proyecto/respuestas_proyecto.txt`](./proyecto/respuestas_proyecto.txt): 30 filas de entrenamiento como ejemplos y 10 filas de 2016 sin las unidades.

| Sobre las mismas 10 filas | MAE |
| --- | --- |
| LLM (Claude Opus 5.5) | 2.7000 |
| Bosque aleatorio | 2.9660 |
| Red neuronal | 3.0690 |
| Siempre 0 (referencia) | 1.9000 |

El LLM sale un poco mejor en estas 10 filas, pero la comparación no es confiable: una sola fila con 18 unidades vendidas, que ninguno predijo, aporta más de la mitad del error de los tres. En 8 de las 10 filas se vendieron 0 unidades, así que "siempre 0" gana sin aprender nada. Además, mi problema se puede plantear así solo a medias: al LLM le faltan el historial de ventas y el producto concreto, y con 30 ejemplos no puede aprender el comportamiento de 3049 productos.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

| Opción | MAE | Costo | Explicabilidad |
| --- | --- | --- | --- |
| Bosque aleatorio | 1.5069 en las 43,845 filas de prueba (2.97 en las 10 de la muestra) | 80.6 s de entrenamiento en CPU; predecir es casi gratis | Buena: `feature_importances_` indica qué columnas pesan más |
| Red neuronal | 1.6706 en las 43,845 filas (3.07 en las 10) | 216.9 s de entrenamiento en CPU; predecir es casi gratis | Difícil: 1793 pesos sin significado directo |
| LLM (Claude Opus 5.5) | 2.70 en las 10 filas; no medible en las 43,845 | Sin entrenamiento, pero unos 224 dólares al mes solo de entrada (613 tokens × 3049 prompts al día × 30 días × $4 por millón), sin contar la salida | Da una explicación en palabras, sin garantía de que sea la razón real |

Usaría el **bosque aleatorio**. Es el de menor error entre los que pude medir sobre las 43,845 filas de 2016 (1.5069 contra 1.6706 de la red), entrena en poco más de un minuto en mi CPU y después predice casi gratis. El LLM ganó en las 10 filas de la muestra (2.70 contra 2.97), pero una diferencia de 0.27 en 10 filas, con una de ellas aportando más de la mitad del error, no prueba nada. Y costaría unos 224 dólares al mes solo de entrada para predecir todos mis productos, con el riesgo de dar números distintos para el mismo caso. Un LLM aportaría más en tareas de texto, que mi proyecto no tiene: mis datos son una tabla.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Tendría que saber cómo se dividieron los datos (la semilla y el tamaño de prueba), si el archivo de datos es exactamente el mismo, los hiperparámetros y la semilla del modelo, la preparación y la versión del código, qué métrica midió cada uno y sobre qué conjunto, y las versiones de las librerías.
