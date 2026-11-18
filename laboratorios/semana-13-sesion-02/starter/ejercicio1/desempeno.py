"""El desempeno del modelo en produccion: une cada prediccion guardada con
la tarifa que se cobro, y calcula el error por semana.

    python desempeno.py
"""

import os
import sys

import pandas as pd

MAE_REFERENCIA = 1.17  # el error en prueba con que se aprobo la version 1

if not os.path.exists("predicciones.jsonl"):
    sys.exit("No hay predicciones.jsonl: el servicio no esta guardando sus predicciones.")

predicciones = pd.read_json("predicciones.jsonl", lines=True)
cobros = pd.read_csv("cobros.csv")

# TODO 4: une predicciones y cobros por id_prediccion (pd.merge), calcula el
# error absoluto de cada viaje, agrega la columna "semana" con el lunes de
# la semana del viaje (fecha_hora_inicio.dt.to_period("W").dt.start_time.dt.date),
# y agrupa por semana: "viajes" (cuantos) y "mae" (el promedio del error).
datos = pd.merge(predicciones, cobros, on="id_prediccion")  #---------cada prediccion con la tarifa que se cobro de verdad en ese viaje
datos["error_absoluto"] = (datos["tarifa_cobrada"] - datos["tarifa_estimada"]).abs()  #---------cuanto se equivoco el modelo en cada viaje
datos["fecha_hora_inicio"] = pd.to_datetime(datos["fecha_hora_inicio"])  #---------para poder sacar la semana del viaje
datos["semana"] = datos["fecha_hora_inicio"].dt.to_period("W").dt.start_time.dt.date  #---------el lunes de la semana del viaje
por_semana = datos.groupby("semana").agg(viajes=("id_prediccion", "count"), mae=("error_absoluto", "mean"))  #---------cuantos viajes y el error promedio por semana
por_semana["mae"] = por_semana["mae"].round(4)

por_semana["veces_la_referencia"] = (por_semana["mae"] / MAE_REFERENCIA).round(2)
print(por_semana.to_string())
