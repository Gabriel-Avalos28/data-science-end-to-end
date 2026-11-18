"""Servicio de prediccion de la tarifa, instrumentado.

Ademas de responder la tarifa, mide cuanto tarda cada prediccion y con
que codigo responde (las metricas del sistema, en /metricas), y guarda
cada prediccion en predicciones.jsonl, para medir despues cuanto se
equivoco el modelo cuando llegue la tarifa cobrada (desempeno.py).

    uvicorn servicio:app --port 8000
"""

import json
import os
import time
import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

from features_taxis import agregar_features
from preparacion_taxis import CATEGORICAS, NUMERICAS, YA_LISTAS

RUTA_MODELO = "modelo_tarifa.joblib"
RUTA_VERSION = "modelo_version.txt"
RUTA_REGISTRO = os.environ.get("RUTA_REGISTRO", "predicciones.jsonl")

modelo = None
version_modelo = None
if os.path.exists(RUTA_MODELO):
    modelo = joblib.load(RUTA_MODELO)
    with open(RUTA_VERSION) as archivo:
        version_modelo = archivo.read().strip()


class Metricas:
    """Las metricas del sistema, en memoria: una latencia y un codigo por
    cada peticion a /predecir."""

    def __init__(self):
        self.latencias_ms = []
        self.codigos = []

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(latencia_ms)
        self.codigos.append(codigo)

    def resumen(self):
        # TODO 2: devuelve un diccionario con "peticiones" (cuantas hubo),
        # "errores_validacion" (cuantas respondieron 422), "errores_servidor"
        # (cuantas respondieron 500 o mas), y "latencia_p50_ms" y
        # "latencia_p95_ms" (np.percentile, redondeadas a 2 decimales).
        # Sin peticiones, las dos latencias son None.

        latencia_p50 = latencia_p95 = None  #---------sin peticiones, las dos latencias quedan en None

        if self.latencias_ms:
            latencia_p50 = round(float(np.percentile(self.latencias_ms, 50)), 2)  #---------p50: la mediana, lo que tarda una peticion tipica
            latencia_p95 = round(float(np.percentile(self.latencias_ms, 95)), 2)  #---------p95: el 95% de las peticiones tarda menos que esto

        errores_validacion = errores_servidor = 0  #---------los contadores empiezan en 0 (sin esto, += 1 falla)
        for codigo in self.codigos:
            if codigo == 422:
                errores_validacion += 1  #---------422: la entrada no cumple el esquema (lo rechaza FastAPI)
            elif codigo >= 500:
                errores_servidor += 1  #---------500 o mas: el servicio fallo por dentro

        return {
            "peticiones": len(self.codigos),  #---------los nombres exactos que piden el TODO y las pruebas
            "errores_validacion": errores_validacion,
            "errores_servidor": errores_servidor,
            "latencia_p50_ms": latencia_p50,
            "latencia_p95_ms": latencia_p95,
        }


metricas = Metricas()
app = FastAPI(title="Tarifa estimada de taxis")


@app.middleware("http")
async def medir(request: Request, call_next):
    """Corre alrededor de cada peticion: lo que esta antes de call_next
    pasa antes de la ruta, y lo que esta despues, cuando ya hay respuesta."""
    if request.url.path != "/predecir":
        return await call_next(request)
    # TODO 1: toma el tiempo antes y despues de call_next con time.perf_counter(),
    # y registra en metricas la latencia en milisegundos y el codigo de la
    # respuesta (respuesta.status_code).
    
    #---------toda peticion http sera capturada por @app.middleware("http") y sera pasada a la funcion medir, que recibe la peticion y call_next, que es la funcion que llama a la ruta correspondiente. Si la ruta no es /predecir, se llama a call_next y se devuelve la respuesta. Si la ruta es /predecir, se toma el tiempo antes y despues de call_next, se calcula la latencia en milisegundos y se registra en metricas junto con el codigo de la respuesta.
    #------el @ es un decorador, atrapa la funcion de abajo

    inicio = time.perf_counter() #----------cuanto tiempo se tarda en responder de esa ruta de esa funcion, no es un tiempo de respuesta final que ve el cliente

    resultado = await call_next(request)  #---------call_next es asincrona: sin await devuelve una coroutine (sin status_code) y no espera a la ruta

    tiempo_respuesta = (time.perf_counter() - inicio) * 1000

    metricas.registrar(tiempo_respuesta, resultado.status_code)
    
    return resultado


class Viaje(BaseModel):
    """Lo que la app sabe de un viaje antes de que empiece."""
    fecha_hora_inicio: datetime
    zona_origen: str
    zona_destino: str
    distancia_km: float | None = None
    pasajeros: float | None = None
    metodo_pago: str | None = None


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_modelo}


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()


@app.post("/predecir")
def predecir(viaje: Viaje):
    datos = agregar_features(pd.DataFrame([viaje.model_dump()]))
    tarifa = round(float(modelo.predict(datos[NUMERICAS + YA_LISTAS + CATEGORICAS])[0]), 2)
    id_prediccion = uuid.uuid4().hex
    # TODO 3: agrega una linea a RUTA_REGISTRO (abre el archivo con "a") con
    # json.dumps de un diccionario con id_prediccion, "momento"
    # (datetime.now().isoformat()), version_modelo, los campos del viaje
    # (viaje.model_dump(mode="json")) y "tarifa_estimada".

    registro_predicciones = {
        "id_prediccion": id_prediccion,  #---------el mismo id que recibe la app: con el se une despues con la tarifa cobrada
        "momento": datetime.now().isoformat(),  #---------cuando se hizo la prediccion
        "version_modelo": version_modelo,  #---------que version del modelo la calculo
        **viaje.model_dump(mode="json"),  #---------los campos del viaje como columnas sueltas (fecha_hora_inicio, zona_origen, ...)
        "tarifa_estimada": tarifa,  #---------lo que predijo el modelo, para compararlo con lo cobrado
    }

    with open(RUTA_REGISTRO, "a") as archivo:  #---------"a": agrega una linea al final, sin borrar las anteriores
        archivo.write(json.dumps(registro_predicciones) + "\n")  #---------una linea JSON por prediccion (formato JSON Lines)

    return {"id_prediccion": id_prediccion, "tarifa_estimada": tarifa, "version_modelo": version_modelo}
