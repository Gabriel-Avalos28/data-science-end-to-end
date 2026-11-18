"""Simula 8 semanas del servicio en produccion con ventas reales de 2016 (el periodo de prueba,
que el modelo no vio al entrenar). Se corre desde psets/proyecto/, despues de
python servicio/exportar_modelo.py:

    python -m monitoreo.simular_ventas

Para cada semana toma 400 dias-producto de la muestra, le pide la prediccion al servicio (con
TestClient: pasa por el middleware y por el registro, como una peticion real) y anota las
unidades que de verdad se vendieron en monitoreo/ventas_reales.csv: lo que el sistema de ventas
reporta al cierre de cada dia. Las primeras 4 semanas las sirve la version en produccion; las
ultimas 4, la version 2 del registro, para simular un cambio de version y ver el monitoreo por
version. Al final manda 5 peticiones mal formadas, como las de una app desactualizada.
"""

import os
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi.testclient import TestClient

from modelado.pipeline_proyecto import RUTA

CARPETA = Path(__file__).resolve().parent
RUTA_REGISTRO = CARPETA / "predicciones.jsonl"
RUTA_VENTAS = CARPETA / "ventas_reales.csv"
DESDE, HASTA = "2016-03-28", "2016-05-22"  #---------las ultimas 8 semanas completas de la muestra (lunes a domingo)
CAMBIO_DE_VERSION = "2016-04-25"          #---------desde este lunes sirve la version 2
POR_SEMANA = 400

#---------empieza con un registro limpio, para que el reporte sea solo de esta simulacion
for ruta in [RUTA_REGISTRO, RUTA_VENTAS]:
    if ruta.exists():
        ruta.unlink()
os.environ["RUTA_REGISTRO"] = str(RUTA_REGISTRO)

from servicio import api  # noqa: E402  (despues de fijar RUTA_REGISTRO)

cliente = TestClient(api.app)
modelo_v1, version_v1 = api.modelo, api.version_modelo
modelo_v2 = mlflow.sklearn.load_model("models:/pronostico-ventas-tiendas/2")

muestra = pd.read_csv(RUTA)
periodo = muestra[(muestra["DATE"] >= DESDE) & (muestra["DATE"] <= HASTA)].copy()
periodo["semana"] = pd.to_datetime(periodo["DATE"]).dt.to_period("W").dt.start_time
dias = periodo.groupby("semana", group_keys=False).apply(lambda s: s.sample(POR_SEMANA, random_state=42))

ventas = []
for fila in dias.sort_values("DATE").itertuples():
    #---------cambio de version a mitad del periodo, como si se hubiera promovido la version 2
    if fila.DATE >= CAMBIO_DE_VERSION:
        api.modelo, api.version_modelo = modelo_v2, "2"
    else:
        api.modelo, api.version_modelo = modelo_v1, version_v1
    peticion = {"fecha": fila.DATE, "item_id": fila.ITEM_ID, "store_id": fila.STORE_ID,
                "precio": fila.SELL_PRICE, "es_dia_de_evento": bool(fila.IS_EVENT_DAY)}
    cliente.post("/predecir", json=peticion)
    #---------la verdad llega por producto, tienda y dia, no por id_prediccion
    ventas.append({"fecha": fila.DATE, "item_id": fila.ITEM_ID, "store_id": fila.STORE_ID, "unidades": fila.UNITS})

for _ in range(5):
    cliente.post("/predecir", json={"fecha": "2016-05-23", "item_id": "FOODS_3_090"})  # sin tienda ni precio: 422

pd.DataFrame(ventas).to_csv(RUTA_VENTAS, index=False)
print(f"{len(ventas)} predicciones del {DESDE} al {HASTA} en {RUTA_REGISTRO.name}, ventas reales en {RUTA_VENTAS.name}")
print("Metricas del sistema:", cliente.get("/metricas").json())
