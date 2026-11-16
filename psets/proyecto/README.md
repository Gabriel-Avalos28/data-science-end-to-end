# [Nombre de tu proyecto]

> Este es el README de tu proyecto: reemplaza cada sección entre corchetes a medida que avanzas. Escríbelo para alguien que clona tu fork sin haber estado en clase y necesita entender qué hace el proyecto y correrlo, no como una nota para ti.

[Un párrafo: qué problema de negocio resuelve, para quién, y qué predice el modelo. Sale de tu propuesta del PSet 1, que está completa en `docs/`.]

**Métrica de éxito:** [la métrica de negocio y el umbral de decisión del PSet 1]

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| [nombre de la fuente] | [qué filas y columnas trae, desde qué fecha] | [la URL, la API o el procedimiento exacto] | `data/raw/[archivo]` |

Los datos no están en el repositorio. Para regenerarlos desde cero: [los comandos, en orden]. Se completa con el PSet 1.

## Cómo correr el pipeline de datos

[Los comandos, en orden y desde qué carpeta, para ir de la fuente al warehouse: levantar la infraestructura, construir la imagen, subir el flow a Kestra, limpiar, cargar y correr dbt. Cada paso dice qué archivo deja y dónde. Se completa con el PSet 2.]

## Cómo entrenar y evaluar el modelo

[Qué notebook corre cada etapa del modelado, en qué orden, y qué necesita levantado (MLflow). Se completa con el PSet 3.]

## Modelo final

[El modelo elegido, su `run_id` en MLflow y la métrica que lo sostiene. El razonamiento completo está en `docs/decision_modelo.md`. Se completa con el PSet 3.]

## Cómo correr el servicio

El servicio responde cuántas unidades de un producto se venden en una tienda en un día. Lo usa quien necesite una predicción que no está en el lote nocturno (ver [`docs/despliegue.md`](docs/despliegue.md)): un tablero, otra aplicación o una simulación de precio. Todo se corre desde `psets/proyecto/`.

**1. Registrar, promover y exportar el modelo.** Con el servidor de MLflow del Codespace levantado (puerto 5050):

```
python -m servicio.registrar_modelo       # entrena y registra una version nueva de pronostico-ventas-tiendas
python -m servicio.promover 1             # pone el alias produccion en la version 1 (y lista todas las versiones)
python servicio/exportar_modelo.py        # exporta la version que tiene el alias produccion
```

`exportar_modelo.py` guarda la versión con el alias `produccion` en `servicio/modelo.joblib`, su número en `servicio/modelo_version.txt`, y en `servicio/referencias.csv` la fecha de lanzamiento y el precio promedio de entrenamiento de cada producto en cada tienda (30,462 series). Ninguno de los tres se sube: se regeneran con estos scripts, que necesitan la muestra de features de la Semana 9. El criterio para promover una versión, y cómo volver a la anterior, está en [`docs/despliegue.md`](docs/despliegue.md#versionado-y-registro).

**2. Construir y levantar el contenedor.** Usa `scikit-learn==1.5.2`, la misma versión con que se entrenó el modelo:

```
docker build -f servicio/Dockerfile -t servicio-demanda .
docker run -p 8000:8000 servicio-demanda
```

Sin Docker: `uvicorn servicio.api:app --port 8000`. La documentación interactiva queda en `/docs`.

**3. Pedir una predicción.** Quien llama manda solo lo que conoce del día a predecir. El servicio calcula el resto de las features (día de la semana, fin de semana, SNAP, categoría, días desde el lanzamiento y precio relativo) con las funciones de `modelado/pipeline_proyecto.py`.

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" \
  -d '{"fecha": "2016-05-09", "item_id": "FOODS_3_090", "store_id": "TX_2", "precio": 1.5, "es_dia_de_evento": false}'
```

```
{"unidades_estimadas":7.45,"serie_conocida":true,"version_modelo":"1"}
```

`version_modelo` es el número de versión del registro que calculó la predicción, y `serie_conocida` dice si el producto tenía referencias (si no, sus features las imputa la preparación). `GET /salud` dice si el modelo está cargado y qué `version_modelo` sirve. Una entrada inválida (sin precio, con un precio negativo, una fecha mal escrita o una tienda fuera de CA, TX o WI) recibe `422` sin llegar al modelo.

**Pruebas.** `pytest psets/proyecto -v`, desde la raíz del fork, corre las pruebas del pipeline (`modelado/test_pipeline_proyecto.py`) y las del servicio (`servicio/test_api.py`), con un modelo y unas referencias de prueba. GitHub Actions las corre en cada push.
