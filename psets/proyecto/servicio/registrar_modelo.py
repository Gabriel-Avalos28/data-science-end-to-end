"""Entrena el modelo final del proyecto, mide su error en 2016 y lo registra como una
version nueva de "pronostico-ventas-tiendas" en el registro de modelos de MLflow.

Se corre desde psets/proyecto/, como modulo, con el servidor de MLflow levantado:

    python -m servicio.registrar_modelo                      # el modelo final: max_depth=12
    python -m servicio.registrar_modelo --max_depth 6        # otra version, para comparar

Cada vez que se corre crea una version nueva: ninguna reemplaza a otra. No pone el alias
"produccion": eso lo decide servicio/promover.py, despues de comparar las metricas.
"""

import argparse
import hashlib
import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import (
    CORTE, RUTA, agregar_features, costo_stock, dividir_por_fecha, modelo_final,
)

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "pronostico-ventas-tiendas"  #---------el nombre del modelo en el registro: el mismo del experimento del proyecto (Semana 11)
EXPERIMENTO = "pronostico-ventas-tiendas-registro"

argumentos = argparse.ArgumentParser(description=__doc__.splitlines()[0])
argumentos.add_argument("--max_depth", type=int, default=12)
argumentos.add_argument("--min_samples_leaf", type=int, default=50)
parametros = argumentos.parse_args()

#---------los mismos datos y la misma division por fecha que en las Semanas 10 y 11
huella_datos = hashlib.md5(open(RUTA, "rb").read()).hexdigest()[:12]
datos = agregar_features(pd.read_csv(RUTA))
X_ent, X_prueba, y_ent, y_prueba = dividir_por_fecha(datos)

mlflow.set_experiment(EXPERIMENTO)
with mlflow.start_run(run_name=f"bosque-depth{parametros.max_depth}-leaf{parametros.min_samples_leaf}"):
    modelo = modelo_final(max_depth=parametros.max_depth, min_samples_leaf=parametros.min_samples_leaf)
    modelo.fit(X_ent, y_ent)
    estimadas = modelo.predict(X_prueba)

    #---------metrica principal: el costo de stock (una unidad que falta cuesta 3 veces una que sobra)
    #---------y el MAE como metrica secundaria, para comparar con las semanas anteriores
    metricas = {
        "costo_stock_prueba": costo_stock(y_prueba, estimadas),
        "mae_prueba": float(np.mean(np.abs(y_prueba - estimadas))),
    }
    mlflow.log_params({**vars(parametros), "fecha_corte": CORTE, "huella_datos": huella_datos, "filas_entrenamiento": len(X_ent)})
    mlflow.log_metrics(metricas)

    #---------guarda el modelo en la corrida y lo registra como version nueva, en un solo paso
    info = mlflow.sklearn.log_model(modelo, "modelo", registered_model_name=NOMBRE)
    version = info.registered_model_version

print(f"Version {version} de {NOMBRE}: max_depth={parametros.max_depth}, min_samples_leaf={parametros.min_samples_leaf}, "
      f"costo de stock {metricas['costo_stock_prueba']:.4f}, MAE {metricas['mae_prueba']:.4f}")
