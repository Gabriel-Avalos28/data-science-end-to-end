"""Exporta la version del modelo que tiene el alias "produccion" y la tabla de referencias
que necesita el servicio. Se corre desde psets/proyecto/, con el servidor de MLflow levantado:

    python servicio/exportar_modelo.py

Deja tres archivos en servicio/, que no se suben al repositorio (se regeneran con este script)
y que la imagen lleva adentro, asi el contenedor no necesita llegar al servidor de MLflow:
- modelo.joblib: el pipeline entrenado (preparacion + bosque aleatorio) de esa version.
- modelo_version.txt: el numero de version del registro que se exporto.
- referencias.csv: fecha_lanzamiento y precio_promedio de cada producto en cada tienda,
  calculados por features_proyecto.py (Semana 9) sobre el archivo completo, el precio
  solo con dias anteriores al corte.
"""

import sys
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))  # para importar modelado/ al correrlo como script

from modelado.pipeline_proyecto import RUTA, referencias_por_serie  # noqa: E402

NOMBRE = "pronostico-ventas-tiendas"
ALIAS = "produccion"

#---------Semana 13: ya no exporta un run_id fijo, sino la version que tiene el alias "produccion"
version = MlflowClient().get_model_version_by_alias(NOMBRE, ALIAS).version
modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{version}")
joblib.dump(modelo, CARPETA / "modelo.joblib")
(CARPETA / "modelo_version.txt").write_text(version)  #---------el numero viaja junto al modelo, dentro de la imagen
print(f"Version {version} de {NOMBRE} ({ALIAS}) guardada en servicio/modelo.joblib y servicio/modelo_version.txt")

referencias = referencias_por_serie(pd.read_csv(RUTA))
referencias.to_csv(CARPETA / "referencias.csv")
print(f"Referencias de {len(referencias)} series guardadas en servicio/referencias.csv")
