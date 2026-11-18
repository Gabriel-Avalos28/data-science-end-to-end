# Monitoreo del modelo de demanda

El modelo `pronostico-ventas-tiendas` (versión 1 en `produccion`, ver [`despliegue.md`](despliegue.md)) se sirve por lotes cada noche y también con el servicio de `servicio/`. Se vigilan dos tipos de métricas:

- **Métricas de sistema:** si el servicio responde, qué tan rápido y con qué errores. Se conocen al instante y salen de `/metricas`.
- **Métricas de modelo:** si las predicciones aciertan. Necesitan las ventas reales, que llegan al cierre de cada día. Mientras tanto se miran indicadores que no necesitan la verdad.

Cada predicción queda en `monitoreo/predicciones.jsonl`, con `id_prediccion`, `momento`, `version_modelo`, la entrada (`fecha`, `item_id`, `store_id`, `precio`, `es_dia_de_evento`) y la salida (`unidades_estimadas`, `serie_conocida`). Las ventas reales llegan por producto, tienda y día, así que la llave para unirlas con la predicción es `(fecha, item_id, store_id)`, y está completa en el registro. [`monitoreo/desempeno.py`](../monitoreo/desempeno.py) hace esa unión.

## Qué se vigila

| Métrica | Tipo | De dónde sale | Cuándo se conoce | Alerta si |
|---|---|---|---|---|
| Latencia p95 de `/predecir` | Sistema | `/metricas` (middleware de `servicio/api.py`) | Al instante | p95 > 100 ms durante 5 minutos (en la simulación, p50 36 ms y p95 44 ms) |
| Errores de servidor (500 o más, incluye 503 sin modelo) | Sistema | `/metricas` | Al instante | Cualquier error: el problema es nuestro |
| Errores de validación (422) | Sistema (calidad de los datos que llegan) | `/metricas` | Al instante | Más del 1% de las peticiones: alguien llama con un formato viejo o roto |
| Modelo cargado | Sistema | `/salud` (`modelo_cargado`, `version_modelo`) | Al instante | `modelo_cargado` es `false`, o `version_modelo` no es la que tiene el alias `produccion` |
| Filas del lote nocturno | Sistema | La tabla que deja el lote | Al terminar el lote | Menos de 30,490 series: el lote quedó incompleto |
| Unidades estimadas promedio por día-producto | Modelo, indirecta (deriva de las predicciones) | `monitoreo/predicciones.jsonl` | El mismo día | Fuera de ±20% de 1.44 (el promedio vendido en entrenamiento) en una semana |
| Fracción de series sin referencias (`serie_conocida = false`) | Modelo, indirecta (calidad de la entrada) | `monitoreo/predicciones.jsonl` | El mismo día | Más del 1%: productos nuevos, con features imputadas |
| Costo de stock por semana y por versión | **Modelo (la métrica del negocio)** | `monitoreo/desempeno.py` (registro + ventas reales) | Al día siguiente (cierre de ventas del día) | Más de 1.25 veces la referencia (3.0457 de la versión 1 en prueba) en una semana |
| MAE por semana y por versión | Modelo | `monitoreo/desempeno.py` | Al día siguiente | Informativo: no decide solo, porque "siempre 0" gana en MAE (Semana 11) |

## Lo que mostró la simulación

`python -m monitoreo.simular_ventas` pasó por el servicio 3,200 días-producto reales de las últimas 8 semanas de 2016 (400 por semana). Las primeras 4 semanas las sirvió la versión 1 y las últimas 4 la versión 2. Después `python -m monitoreo.desempeno` las unió con las ventas reales:

| Semana | Versión | Costo de stock | Veces la referencia | Promedio estimado | Promedio vendido |
|---|---|---|---|---|---|
| 2016-03-28 | 1 | 2.6568 | 0.87 | 1.39 | 1.34 |
| 2016-04-04 | 1 | 3.2952 | 1.08 | 1.38 | 1.51 |
| 2016-04-11 | 1 | 2.9411 | 0.97 | 1.35 | 1.35 |
| 2016-04-18 | 1 | 2.9022 | 0.95 | 1.39 | 1.43 |
| 2016-04-25 | 2 | 3.2934 | 1.08 | 1.41 | 1.41 |
| 2016-05-02 | 2 | 3.2290 | 1.06 | 1.54 | 1.48 |
| 2016-05-09 | 2 | **4.1095** | **1.35** | 1.51 | **1.88** |
| 2016-05-16 | 2 | 3.0623 | 1.01 | 1.36 | 1.42 |

Métricas de sistema: 3,205 peticiones, 5 errores de validación (las peticiones mal formadas a propósito), 0 errores de servidor, p50 de 36 ms y p95 de 44 ms.

La semana del 9 de mayo cruza el umbral del costo (1.35 veces la referencia), pero ningún indicador sin la verdad lo anticipó: el promedio estimado (1.51) estaba dentro de ±20% de 1.44. Lo que pasó es que esa semana **se vendió más** (1.88 por día-producto). El modelo subestimó, que es el error caro. Solo las ventas reales lo muestran.

En promedio, la versión 1 tuvo un costo de 2.95 en sus 4 semanas y la versión 2 de 3.42. Pero cada una sirvió semanas distintas, así que esa diferencia mezcla la versión con el período. Para comparar dos versiones de verdad, las dos tienen que predecir los mismos días.
