"""El desempeno del modelo en produccion: une cada prediccion del registro con las unidades
que de verdad se vendieron, y calcula la metrica de la Semana 11 por semana y por version.
Se corre desde psets/proyecto/:

    python -m monitoreo.desempeno

La verdad de este proyecto llega rapido: las ventas de un dia se conocen al cierre de ese dia.
Mientras tanto (o para las predicciones de dias futuros), el reporte tambien muestra dos
metricas que se calculan sin la verdad, en el momento de predecir.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import costo_stock

CARPETA = Path(__file__).resolve().parent
COSTO_REFERENCIA = 3.0457     # costo de stock de la version 1 en la prueba de 2016 (Semana 13, sesion 1)
PROMEDIO_REFERENCIA = 1.44    # unidades vendidas por dia-producto en entrenamiento (antes de 2016): la escala de las predicciones
LLAVE = ["fecha", "item_id", "store_id"]


def unir_con_la_verdad(predicciones, ventas):
    """Cada prediccion con las unidades vendidas ese dia, en ese producto y esa tienda."""
    #---------la llave es (fecha, producto, tienda): es lo que reporta el sistema de ventas
    datos = pd.merge(predicciones, ventas, on=LLAVE, how="inner")
    datos["semana"] = pd.to_datetime(datos["fecha"]).dt.to_period("W").dt.start_time.dt.date  #---------el lunes de la semana
    return datos


def desempeno_por_semana(datos):
    """MAE y costo de stock por semana y por version: las metricas de MODELO (necesitan la verdad)."""
    filas = []
    for (semana, version), grupo in datos.groupby(["semana", "version_modelo"]):
        filas.append({
            "semana": semana, "version": version, "predicciones": len(grupo),
            "mae": round(float(np.mean(np.abs(grupo["unidades"] - grupo["unidades_estimadas"]))), 4),
            "costo_stock": round(float(costo_stock(grupo["unidades"], grupo["unidades_estimadas"])), 4),  #---------la metrica del negocio (Semana 11)
        })
    tabla = pd.DataFrame(filas).set_index(["semana", "version"])
    tabla["veces_la_referencia"] = (tabla["costo_stock"] / COSTO_REFERENCIA).round(2)
    return tabla


def indicadores_sin_verdad(predicciones):
    """Lo que se puede vigilar el mismo dia, sin esperar las ventas: la escala de las predicciones
    (deriva de las predicciones) y que fraccion de las peticiones son de productos sin referencias."""
    predicciones = predicciones.copy()
    predicciones["semana"] = pd.to_datetime(predicciones["fecha"]).dt.to_period("W").dt.start_time.dt.date
    tabla = predicciones.groupby(["semana", "version_modelo"]).agg(
        promedio_estimado=("unidades_estimadas", "mean"),           #---------si se mueve mucho, cambio el modelo o los datos que llegan
        series_desconocidas=("serie_conocida", lambda s: 1 - s.mean()),  #---------productos nuevos: sus features se imputan
    ).round(4)
    return tabla


if __name__ == "__main__":
    predicciones = pd.read_json(CARPETA / "predicciones.jsonl", lines=True, dtype={"fecha": str, "version_modelo": str})
    ventas = pd.read_csv(CARPETA / "ventas_reales.csv")
    datos = unir_con_la_verdad(predicciones, ventas)
    print(f"{len(predicciones)} predicciones en el registro, {len(datos)} unidas con sus ventas reales\n")
    print("Metricas de modelo, por semana y version (necesitan la verdad):")
    print(desempeno_por_semana(datos).to_string())
    print(f"\nMetricas sin la verdad, por semana y version (referencia: {PROMEDIO_REFERENCIA} unidades por dia-producto):")
    print(indicadores_sin_verdad(predicciones).to_string())
    print("\nPromedio real vendido por semana (para comparar con el estimado):")
    print(datos.groupby("semana")["unidades"].mean().round(4).to_string())
