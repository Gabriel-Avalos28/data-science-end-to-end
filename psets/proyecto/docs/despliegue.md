# Despliegue: por lotes, una vez por noche

**Decisión:** el modelo de demanda se sirve **por lotes**. Cada noche, un script estima las unidades que va a vender cada producto en cada tienda en los días siguientes y deja las estimaciones en una tabla. El sistema de reabastecimiento lee esa tabla por la mañana: nadie espera a que el modelo responda.

El modelo es el bosque aleatorio elegido en la Semana 11 (`run_id` `853a6190c5b64f2cb5aa9333c55445c0`: MAE de prueba 1.5069, costo de stock 3.05), con la preparación y las features de [`modelado/pipeline_proyecto.py`](../modelado/pipeline_proyecto.py).

## Qué decisión habilita y cuánto puede esperar

La predicción decide **cuántas unidades de cada producto pedir o reponer en cada tienda**. La usan el sistema de reabastecimiento y los encargados de inventario de cada tienda, que arman el pedido una vez al día. Lo más caro es quedarse corto: una unidad que falta cuesta unas 3 veces una que sobra (la métrica `costo_stock` de la Semana 11).

Como el pedido se arma una vez al día, la predicción puede esperar **horas, no milisegundos**: basta con que esté lista antes de que abra la ventana de pedidos. Si el lote corre a las 00:00 y los pedidos se arman desde las 06:00, el lote tiene **6 horas** de margen, y la estimación de un día puede tener hasta **24 horas** de antigüedad sin que cambie la decisión. Nadie está esperando frente a una pantalla.

## Cuántos casos hay y si se conocen de antemano

Los casos son finitos y se conocen todos de antemano: **10 tiendas × 3,049 productos = 30,490 series** (producto en tienda). El catálogo y las tiendas no cambian de un día para otro, y la fecha a predecir es siempre "mañana" (o los próximos 28 días, el horizonte del M5).

| Lote | Filas | Tiempo de predicción medido |
|---|---|---|
| Mañana, todas las series | 30,490 | 0.16 s |
| Próximos 28 días, todas las series | 853,720 | 4.5 s |
| Una sola fila (como haría un servicio en línea) | 1 | 25.7 ms |

Los tiempos son del bosque final entrenado con la muestra, en el Codespace. Predecir todo el horizonte de 28 días tarda menos de 5 segundos: el lote usa una fracción mínima de su ventana de 6 horas.

## Qué datos necesitan las features al predecir

Todas las features se conocen **antes** del día que se predice; ninguna depende de algo que solo exista en el momento:

| Feature | De dónde sale al predecir | ¿Disponible la noche anterior? |
|---|---|---|
| `SELL_PRICE` | El precio planificado del producto (Walmart fija los precios por semana) | Sí |
| `WEEKDAY`, `IS_WEEKEND` | La fecha a predecir | Sí |
| `IS_EVENT_DAY`, `IS_SNAP` | El calendario de eventos y el calendario SNAP de cada estado, publicados con anticipación | Sí |
| `CAT_ID`, `STORE_ID` | El catálogo | Sí |
| `dias_desde_lanzamiento` | La fecha a predecir menos `fecha_lanzamiento`, una tabla por serie | Sí |
| `precio_relativo` | `SELL_PRICE` dividido para `precio_promedio`, una tabla fija calculada solo con entrenamiento | Sí (vacío para productos nuevos: lo imputa la preparación) |

Las columnas que no existen al predecir (`UNITS`, `REVENUE`, `IS_SALES_SPIKE`, `IS_SUSPECTED_STOCKOUT`) ya estaban fuera del modelo, y una prueba lo comprueba (`test_el_objetivo_y_las_fugas_no_llegan_al_modelo`). El modelo tampoco usa ventas de ayer (lags): la estimación no cambia por lo que se venda durante el día, así que no hay nada que un servicio en línea pudiera ver y el lote no.

## Qué se pierde con el patrón descartado (en línea)

Por lotes se pierde:

- **Reaccionar el mismo día.** Si cambia un precio o se agrega una promoción o un producto después de que corrió el lote, la tabla no lo refleja hasta la noche siguiente. Se mitiga volviendo a correr el lote, que tarda segundos.
- **Preguntas fuera de la tabla.** Un "¿cuánto vendería si bajo el precio a 2.50?" o un producto que no estaba en el catálogo no tienen fila que leer. Para eso existe el servicio de la Sesión 2, como complemento del lote y no como reemplazo.
- **Cálculo que nadie usa.** Se estiman las 30,490 series aunque muchas no se pidan ese día (cerca del 68% de los días-producto venden 0). A 0.16 s por noche, el desperdicio es irrelevante.

En línea, en cambio, costaría más sin ganar nada para esta decisión: un servidor encendido todo el día, tablas de `fecha_lanzamiento` y `precio_promedio` disponibles en cada petición, y 30,490 llamadas de ~26 ms cada una (unos 13 minutos en serie) contra 0.16 s de un solo lote. Esa velocidad solo serviría si alguien esperara la respuesta en milisegundos, y aquí nadie espera.
