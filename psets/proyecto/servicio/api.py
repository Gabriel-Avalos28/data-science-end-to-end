"""Servicio de prediccion de la demanda: cuantas unidades de un producto se venden en una
tienda en un dia. Corre desde psets/proyecto/, despues de python servicio/exportar_modelo.py:

    uvicorn servicio.api:app --port 8000

Quien llama manda lo que conoce del dia que quiere predecir: la fecha, el producto, la tienda,
el precio y si es dia de evento. El servicio calcula el resto de las features (dia de la semana,
fin de semana, SNAP, categoria, dias desde el lanzamiento y precio relativo) con las funciones de
modelado/pipeline_proyecto.py, las mismas con que se entreno el modelo.
"""

import json
import os
import time
import uuid
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import SERIE, columnas_modelo, preparar_peticiones

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))
RUTA_REFERENCIAS = Path(os.environ.get("RUTA_REFERENCIAS", CARPETA / "referencias.csv"))
RUTA_VERSION = CARPETA / "modelo_version.txt"  #---------Semana 13: el numero de version del registro, exportado junto al modelo
#---------Semana 13, sesion 2: cada prediccion queda en monitoreo/predicciones.jsonl (no se sube a git)
RUTA_REGISTRO = Path(os.environ.get("RUTA_REGISTRO", CARPETA.parent / "monitoreo" / "predicciones.jsonl"))

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



class Metricas:
    """Las metricas del sistema, en memoria: una latencia y un codigo por cada peticion a /predecir."""

    def __init__(self):
        self.latencias_ms = []
        self.codigos = []

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(latencia_ms)
        self.codigos.append(codigo)

    def resumen(self):
        latencia_p50 = latencia_p95 = None  #---------sin peticiones, no hay latencias que resumir
        if self.latencias_ms:
            latencia_p50 = round(float(np.percentile(self.latencias_ms, 50)), 2)  #---------la peticion tipica
            latencia_p95 = round(float(np.percentile(self.latencias_ms, 95)), 2)  #---------las lentas: el 95% tarda menos que esto
        return {
            "peticiones": len(self.codigos),
            "errores_validacion": sum(codigo == 422 for codigo in self.codigos),  #---------la entrada no cumplio el esquema: el problema es de quien llama
            "errores_servidor": sum(codigo >= 500 for codigo in self.codigos),  #---------el servicio fallo por dentro (o no tenia modelo, 503): el problema es nuestro
            "latencia_p50_ms": latencia_p50,
            "latencia_p95_ms": latencia_p95,
        }


metricas = Metricas()
app = FastAPI(title="Demanda diaria por producto y tienda (Walmart M5)")


@app.middleware("http")
async def medir(request: Request, call_next):
    """Mide la latencia y el codigo de cada peticion a /predecir."""
    if request.url.path != "/predecir":
        return await call_next(request)
    inicio = time.perf_counter()  #---------el reloj arranca antes de la ruta
    try:
        respuesta = await call_next(request)  #---------await: espera a que la ruta responda
    except Exception:
        #---------si la ruta lanza una excepcion no hay respuesta todavia (el 500 lo arma despues Starlette):
        #---------la cuento como error de servidor y la dejo seguir
        metricas.registrar((time.perf_counter() - inicio) * 1000, 500)
        raise
    metricas.registrar((time.perf_counter() - inicio) * 1000, respuesta.status_code)  #---------latencia en milisegundos y codigo
    return respuesta


def guardar_prediccion(registro):
    """Agrega una linea JSON al registro de predicciones."""
    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)  #---------en el contenedor la carpeta monitoreo/ puede no existir
    with open(RUTA_REGISTRO, "a") as archivo:  #---------"a": agrega al final sin borrar
        archivo.write(json.dumps(registro) + "\n")


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


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()  #---------metricas de SISTEMA: volumen, errores y latencia


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
    unidades = round(float(modelo.predict(datos[columnas_modelo()])[0]), 2)
    serie_conocida = bool(datos["fecha_lanzamiento"].notna().iloc[0])
    id_prediccion = uuid.uuid4().hex  #---------identificador unico de esta prediccion

    #---------el registro guarda la entrada completa: (fecha, item_id, store_id) es la llave para unirla
    #---------despues con las ventas reales, que llegan por producto, tienda y dia (no por id_prediccion)
    guardar_prediccion({
        "id_prediccion": id_prediccion,
        "momento": datetime.now().isoformat(),
        "version_modelo": version_modelo,
        **peticion.model_dump(mode="json"),
        "unidades_estimadas": unidades,
        "serie_conocida": serie_conocida,
    })

    return {
        "id_prediccion": id_prediccion,
        "unidades_estimadas": unidades,
        "serie_conocida": serie_conocida,
        "version_modelo": version_modelo,  #---------cada prediccion dice que version la calculo
    }
