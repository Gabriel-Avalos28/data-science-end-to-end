"""Exporta el modelo final desde MLflow y la tabla de referencias que necesita el servicio.

Se corre desde psets/proyecto/, con el servidor de MLflow del Codespace levantado:

    python servicio/exportar_modelo.py

Deja dos archivos en servicio/, que no se suben al repositorio (se regeneran con este script):
- modelo.joblib: el pipeline entrenado (preparacion + bosque aleatorio) del run final.
- referencias.csv: fecha_lanzamiento y precio_promedio de cada producto en cada tienda,
  calculados por features_proyecto.py (Semana 9) sobre el archivo completo, el precio
  solo con dias anteriores al corte.
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))  # para importar modelado/ al correrlo como script

from modelado.pipeline_proyecto import RUTA, referencias_por_serie  # noqa: E402

RUN_ID = "853a6190c5b64f2cb5aa9333c55445c0"  # bosque aleatorio, candidato = final (Semana 11)

modelo = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/modelo")
joblib.dump(modelo, CARPETA / "modelo.joblib")
(CARPETA / "modelo_info.json").write_text(json.dumps({"run_id": RUN_ID}))
print(f"Modelo del run {RUN_ID} guardado en servicio/modelo.joblib")

referencias = referencias_por_serie(pd.read_csv(RUTA))
referencias.to_csv(CARPETA / "referencias.csv")
print(f"Referencias de {len(referencias)} series guardadas en servicio/referencias.csv")
