"""Servicio de prediccion de la demanda: cuantas unidades de un producto se venden en una
tienda en un dia. Corre desde psets/proyecto/, despues de python servicio/exportar_modelo.py:

    uvicorn servicio.api:app --port 8000

Quien llama manda lo que conoce del dia que quiere predecir: la fecha, el producto, la tienda,
el precio y si es dia de evento. El servicio calcula el resto de las features (dia de la semana,
fin de semana, SNAP, categoria, dias desde el lanzamiento y precio relativo) con las funciones de
modelado/pipeline_proyecto.py, las mismas con que se entreno el modelo.
"""

import os
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import SERIE, columnas_modelo, preparar_peticiones

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))
RUTA_REFERENCIAS = Path(os.environ.get("RUTA_REFERENCIAS", CARPETA / "referencias.csv"))
RUTA_VERSION = CARPETA / "modelo_version.txt"  #---------Semana 13: el numero de version del registro, exportado junto al modelo

# El modelo y las referencias se cargan una sola vez, al arrancar, no en cada peticion.
modelo, version_modelo = None, None
if RUTA_MODELO.exists():
    modelo = joblib.load(RUTA_MODELO)
if RUTA_VERSION.exists():
    version_modelo = RUTA_VERSION.read_text().strip()  #---------el contenedor sabe que version sirve sin llegar al servidor de MLflow

referencias = pd.DataFrame(columns=["fecha_lanzamiento", "precio_promedio"],
                           index=pd.MultiIndex.from_arrays([[], []], names=SERIE))
if RUTA_REFERENCIAS.exists():
    referencias = pd.read_csv(RUTA_REFERENCIAS, index_col=SERIE)

app = FastAPI(title="Demanda diaria por producto y tienda (Walmart M5)")


class Peticion(BaseModel):
    """Lo que sabe quien pide la prediccion, antes del dia que quiere predecir."""
    fecha: date = Field(description="El dia a predecir", examples=["2016-05-23"])
    item_id: str = Field(pattern=r"^(FOODS|HOBBIES|HOUSEHOLD)_\d+_\d+$", examples=["FOODS_3_090"])
    store_id: str = Field(pattern=r"^(CA|TX|WI)_\d+$", examples=["CA_3"])
    precio: float = Field(gt=0, description="Precio de venta planificado para ese dia", examples=[1.25])
    es_dia_de_evento: bool = Field(default=False, description="Feriado o evento del calendario")


@app.get("/salud")
def salud():
    return {
        "estado": "ok",
        "modelo_cargado": modelo is not None,
        "version_modelo": version_modelo,  #---------la version del registro "pronostico-ventas-tiendas" que esta sirviendo
        "series_con_referencia": len(referencias),
    }


@app.post("/predecir")
def predecir(peticion: Peticion):
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no esta cargado: corre servicio/exportar_modelo.py")

    fila = pd.DataFrame([{
        "DATE": peticion.fecha.isoformat(),
        "ITEM_ID": peticion.item_id,
        "STORE_ID": peticion.store_id,
        "SELL_PRICE": peticion.precio,
        "IS_EVENT_DAY": peticion.es_dia_de_evento,
    }])
    datos = preparar_peticiones(fila, referencias)
    unidades = float(modelo.predict(datos[columnas_modelo()])[0])

    return {
        "unidades_estimadas": round(unidades, 2),
        "serie_conocida": bool(datos["fecha_lanzamiento"].notna().iloc[0]),
        "version_modelo": version_modelo,  #---------cada prediccion dice que version la calculo
    }
