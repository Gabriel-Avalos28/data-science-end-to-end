# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

Es cuando un modelo tiene una accuracy alta pero no sirve para lo que importa. Si una clase es muy poco frecuente, predecir siempre la clase mayoritaria ya da una accuracy altísima sin detectar ningún caso de la clase rara. En su lugar recomienda precision, recall y F1 para las clases minoritarias, y en la evaluación de LLMs recall, precision o PR-AUC.

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

Da un ejemplo de negocio y uno de LLMs:

- **Business Intelligence:** un gerente fija el objetivo de aumentar 20% la duración de las sesiones en una agencia de noticias online. El equipo alarga las introducciones y agrega relleno: el número sube, pero baja la calidad y el valor para el usuario. También menciona una plataforma que esconde el botón de "Cancelar suscripción" para bajar artificialmente la pérdida de clientes.
- **LLMs:** un modelo de resumen de noticias optimizado en exceso para ROUGE, que premia las coincidencias exactas de n-gramas con resúmenes de referencia, empieza a copiar bloques de texto y a usar palabras de moda en lugar de generalizar. Obtiene una buena puntuación, pero el resumen es peor.

Al elegir un modelo por una sola métrica pasa lo mismo: si esa métrica se vuelve el objetivo, se puede salir bien en ese número sin cumplir el propósito real. Lo vi en mi proyecto: en MAE gana predecir siempre 0 unidades, un modelo que no sirve para decidir cuánto stock tener.

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Cambiaría bastante. Con 10 divisiones distintas, el error de la red neuronal fue de 1.0954 a 1.2477 (desviación 0.0456), y la ventaja sobre el gradient boosting bajó de 0.1058 en la división 42 a 0.0223 en promedio. En el registro casi no se veía, porque cada corrida guarda un solo número de una sola división; la única pista era el gradient boosting del Ejercicio 1, que pasó de 1.2754 a 1.1867 al cambiar la semilla.

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Esperaba que fueran parecidos, pero no iguales, con unos 0.05 de variación hacia arriba o abajo, como vi la sesión pasada al cambiar la división. Salieron entre 1.16 y 1.28, con promedio 1.2288 y desviación 0.0361. El 1.2754 de una sola división estaba cerca del pliegue más alto, así que era un número algo pesimista. El gradient boosting le ganó al bosque en los 5 pliegues.

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

Mayor, porque el modelo solo ve el pasado y tiene que predecir viajes que todavía no han pasado, y los primeros pliegues entrenan con pocos datos. Con pliegues al azar, entrenamiento y validación están mezclados en el tiempo, y eso facilita la predicción. Así salió: 1.3099 contra 1.2288, y con más variación (0.0693 contra 0.0361).

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

El que deje fuera JFK Airport, porque es la zona más distinta: tarifa promedio de 65.02 dólares contra 20 a 30 en las demás, 23.7 km en promedio y 79% de tarifa fija. Si no la ve al entrenar, no puede aprender cómo se comporta. Fue justo el peor: ese pliegue (JFK y Brooklyn) dio 6.43, mientras los otros cuatro quedaron entre 1.19 y 1.31.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El porcentaje de viajes subestimados en más de 20%, porque mide exactamente lo que dispara el reclamo. El MAE (1.2754) y el RMSE (1.9213) no distinguen si el error es hacia arriba o hacia abajo, ni lo miden en proporción a la tarifa. En prueba salió 4.37%, es decir, unos 26 de los 595 viajes generarían reclamo.

**Lo que confirmó la tabla, esquema por esquema:**

| Esquema | Modelo | Promedio | Desviación | Peor pliegue |
| --- | --- | --- | --- | --- |
| Aleatorio | gradient boosting | 1.2288 | 0.0361 | 1.28 |
| Aleatorio | bosque aleatorio | 1.2880 | 0.0291 | 1.34 |
| Temporal | gradient boosting | 1.3099 | 0.0693 | 1.40 |
| Por zona | gradient boosting | 2.2996 | 2.0648 | 6.43 |

El esquema cambia la conclusión sobre qué tan bueno es el modelo: el error al azar es optimista si el modelo tiene que predecir el futuro, y por zona se ve que falla mucho en una zona que nunca vio.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

Accuracy de 0.9508 y recall de 0: acierta el 95% de los pagos sin detectar ni un solo fraude.

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

La accuracy sube solo 2.04 puntos (de 0.9508 a 0.9712), mientras que el recall pasa de 0 a 0.4610: detecta casi la mitad de los fraudes. El recall describe mejor lo que gana la empresa, porque cada fraude detectado es un contracargo que se evita; la accuracy casi no se mueve porque el 95% de los pagos ya eran legítimos.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con el umbral 0.5 deja pasar 159 de los 295 fraudes, con 14 falsas alarmas. El paso 5 eligió el umbral 0.1: deja pasar 122 fraudes y da 222 falsas alarmas. Aunque hay muchas más falsas alarmas, el costo total baja de 6402 a 5546 dólares (13.4% menos), porque cada fraude cuesta 40 dólares y cada falsa alarma solo 3.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

Sube. Con 20 dólares por falsa alarma ya no conviene revisar tantos pagos, y el umbral pasa de 0.1 a 0.5, con un costo total de 6640 dólares. Lo comprobé en la última celda del notebook, sin tocar la constante que usa la comprobación.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

**Mi entrega:** [`proyecto/validacion_proyecto.ipynb`](./proyecto/validacion_proyecto.ipynb), con la tabla en [`proyecto/validacion_proyecto.csv`](./proyecto/validacion_proyecto.csv) y la copia de mi [`pipeline_proyecto.py`](./proyecto/pipeline_proyecto.py) al lado.

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

Predigo las unidades de cada producto por tienda para saber cuánto stock tener. Lo más caro es **subestimar**: si faltan unidades pierdo la venta y su margen, y el cliente puede irse a otra tienda. Sobreestimar solo deja unidades que normalmente se venden después, con el costo de almacenarlas. Supongo, de forma aproximada, que **una unidad que falta cuesta 3 veces una unidad que sobra**. El 3 es una suposición mía, no un dato de la empresa. La métrica `costo_stock` del notebook multiplica por 3 las unidades que faltan y por 1 las que sobran.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

Temporal, porque predigo las ventas de días futuros: cada pliegue entrena con días anteriores y valida con los siguientes, y parto por días para que un mismo día nunca quede a los dos lados. No uso validación por grupo porque el modelo no usa `ITEM_ID` y lo que quiero es predecir el futuro de productos que ya existen. Tampoco estratificada, porque es una regresión. Solo uso datos anteriores a 2016, para no tocar mi conjunto de prueba.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

| Esquema | Modelo | MAE promedio | Desviación | Costo promedio | Desviación |
| --- | --- | --- | --- | --- | --- |
| Temporal | siempre 0 | 1.4321 | 0.1857 | 4.2963 | 0.5571 |
| Temporal | bosque aleatorio (final) | 1.6855 | 0.1844 | 3.2217 | 0.4077 |
| Temporal | red neuronal (segundo) | 1.8832 | 0.2290 | 3.4773 | 0.4127 |
| Al azar | siempre 0 | 1.4370 | 0.0068 | 4.3111 | 0.0204 |
| Al azar | bosque aleatorio (final) | 1.5585 | 0.0062 | 3.1172 | 0.0179 |
| Al azar | red neuronal (segundo) | 1.6727 | 0.0293 | 3.3564 | 0.0123 |

El orden de los modelos no cambió, pero sí cuánto puedo confiar en ellos. Al azar, el bosque parecía equivocarse 1.5585, con una desviación de solo 0.0062; con el esquema temporal se equivoca 1.6855, con una desviación 30 veces mayor (0.1844). El esquema al azar era optimista y escondía que el error depende mucho del periodo: los pliegues temporales del bosque van de 1.9456 a 1.4418.

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

En el esquema temporal casi igual: la ventaja del bosque en MAE es de 0.1977 y la desviación entre pliegues del bosque es 0.1844. Pero esa variación es sobre todo del periodo y afecta a los dos modelos a la vez; comparando pliegue por pliegue, el bosque gana en los 5, con una ventaja que varía poco (desviación 0.0552). En costo también gana los 5 (0.2556 de ventaja). Sí, sigo eligiendo el bosque. Una salvedad: la red neuronal llegó al límite de 200 iteraciones en varios pliegues sin terminar de converger, así que con más iteraciones podría mejorar un poco, a cambio de tardar aún más que los 553 segundos que ya tomó en el esquema temporal.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

Elegí el costo de stock, con cada unidad faltante costando 3 veces una sobrante. El MAE trata igual los dos errores, y por eso dice que "siempre 0" es lo mejor (1.4321 contra 1.6855 del bosque, en el esquema temporal). El costo dice lo contrario: "siempre 0" es el peor (4.2963), porque nunca tendría stock para las ventas reales, y el bosque baja el costo a 3.2217, un 25% menos. Es la justificación que me faltaba para no quedarme con la línea base.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La tarifa se podría calcular de noche, de forma aproximada: casi todo depende de la zona de origen, la de destino, la hora y el tipo de día, que son combinaciones limitadas, y se guardaría una tabla para consultarla al instante. El fraude no, porque depende de cosas que solo existen en el momento del pago, como el monto, si la tarjeta es nueva y los intentos fallidos. Ese modelo tiene que responder en línea y antes de aprobar el pago, en menos de un segundo aproximadamente, para que el pasajero no note la espera.
