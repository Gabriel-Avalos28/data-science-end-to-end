"""Pipeline del proyecto: unidades vendidas por producto, tienda y dia (Walmart, M5).

Es el mismo pipeline de la Semana 9, Sesion 2 (y la copia que usaron los notebooks
de las Semanas 10 y 11), reorganizado en funciones para que lo puedan importar las
pruebas, el lote de prediccion y el servicio sin leer el archivo de datos:

    from modelado.pipeline_proyecto import agregar_features, columnas_modelo, preparacion

Importar este modulo no lee nada. Solo al correrlo como script carga la muestra,
divide por fecha y prepara, igual que antes:

    python psets/proyecto/modelado/pipeline_proyecto.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# La muestra del 1% de obt_sales.csv.gz que guarda features_proyecto.py de la Semana 9,
# Sesion 1, con fecha_lanzamiento y precio_promedio ya calculados. No esta en el
# repositorio (*.csv.gz esta en .gitignore).
RUTA = Path(__file__).resolve().parents[3] / "laboratorios" / "semana-09-sesion-01" / "plantilla-features" / "muestra_con_features.csv.gz"

OBJETIVO = "UNITS"  # unidades vendidas de un producto en una tienda en un dia
NUMERICAS = ["SELL_PRICE", "precio_relativo", "dias_desde_lanzamiento"]  # se imputan y se escalan
CATEGORICAS = ["CAT_ID", "STORE_ID", "WEEKDAY"]  # se codifican con one-hot
YA_LISTAS = ["IS_WEEKEND", "IS_EVENT_DAY", "IS_SNAP"]  # ya son 1/0, pasan sin cambios
# Fuera: REVENUE, IS_SALES_SPIKE e IS_SUSPECTED_STOCKOUT no existen al predecir (fuga),
# e ITEM_ID tiene 3049 productos, con one-hot serian 3049 columnas.
FUGAS = ["REVENUE", "IS_SALES_SPIKE", "IS_SUSPECTED_STOCKOUT"]

SERIE = ["ITEM_ID", "STORE_ID"]  # una serie es un producto en una tienda
CORTE = "2016-01-01"  # entrenamiento antes, prueba desde esta fecha

# Metrica del negocio (Semana 11, Sesion 2): una unidad que falta cuesta 3 veces una que sobra.
COSTO_FALTANTE = 3
COSTO_SOBRANTE = 1


def columnas_modelo():
    """Las columnas que recibe el modelo, en el orden con que se entreno."""
    return NUMERICAS + CATEGORICAS + YA_LISTAS


def fecha_lanzamiento_por_serie(ventas):
    """Primer dia a la venta de cada producto en cada tienda (sin los dias antes del lanzamiento)."""
    vendiendo = ventas[~ventas["IS_PRE_LAUNCH"].astype(bool)]
    return vendiendo.groupby(SERIE)["DATE"].min().rename("fecha_lanzamiento")


def precio_promedio_por_serie(ventas, corte=CORTE):
    """Precio promedio de cada producto en cada tienda, solo con dias anteriores al corte:
    si entraran dias de prueba, se colaria informacion del futuro (data leakage)."""
    vendiendo = ventas[~ventas["IS_PRE_LAUNCH"].astype(bool)]
    entrenamiento = vendiendo[vendiendo["DATE"] < corte]
    return entrenamiento.groupby(SERIE)["SELL_PRICE"].mean().rename("precio_promedio")


def agregar_features(datos):
    """Devuelve una copia con las 2 features del proyecto y las banderas en 1/0.

    Necesita DATE, fecha_lanzamiento, SELL_PRICE y precio_promedio (las dos columnas de
    referencia salen de fecha_lanzamiento_por_serie y precio_promedio_por_serie).
    """
    datos = datos.copy()
    # dias_desde_lanzamiento: cuantos dias lleva el producto a la venta en esa tienda.
    datos["dias_desde_lanzamiento"] = (pd.to_datetime(datos["DATE"]) - pd.to_datetime(datos["fecha_lanzamiento"])).dt.days
    # precio_relativo: precio del dia / precio promedio del producto (0.9 = 10% mas barato).
    # Queda vacio si el producto no tiene precio de entrenamiento: lo imputa la preparacion.
    datos["precio_relativo"] = datos["SELL_PRICE"] / datos["precio_promedio"]
    datos[YA_LISTAS] = datos[YA_LISTAS].astype(int)  # vienen como True/False, las paso a 1/0
    return datos


def dividir_por_fecha(datos, corte=CORTE):
    """Entrenamiento antes del corte y prueba desde el corte: el modelo predice ventas futuras."""
    es_entrenamiento = datos["DATE"] < corte
    X = datos[columnas_modelo()]
    y = datos[OBJETIVO]
    return X[es_entrenamiento], X[~es_entrenamiento], y[es_entrenamiento], y[~es_entrenamiento]


def preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


def modelo_final():
    """El bosque aleatorio elegido en la Semana 11 (run_id 853a6190c5b64f2cb5aa9333c55445c0)."""
    return Pipeline([
        ("preparar", preparacion()),
        ("modelo", RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=50, n_jobs=-1, random_state=42)),
    ])


def costo_stock(y_real, y_estimada):
    """Costo promedio por fila, en unidades equivalentes de stock sobrante."""
    diferencia = np.asarray(y_real) - np.asarray(y_estimada)
    return np.mean(COSTO_FALTANTE * np.maximum(diferencia, 0) + COSTO_SOBRANTE * np.maximum(-diferencia, 0))


if __name__ == "__main__":
    datos = agregar_features(pd.read_csv(RUTA))
    print("Filas y columnas del archivo:", datos.shape)

    X_ent, X_prueba, y_ent, y_prueba = dividir_por_fecha(datos)
    es_entrenamiento = datos["DATE"] < CORTE
    print("Entrenamiento: fechas", datos.loc[es_entrenamiento, "DATE"].min(), "a", datos.loc[es_entrenamiento, "DATE"].max())
    print("Prueba: fechas", datos.loc[~es_entrenamiento, "DATE"].min(), "a", datos.loc[~es_entrenamiento, "DATE"].max())
    print("Vacios en entrenamiento:", X_ent.isna().sum()[lambda s: s > 0].to_dict())
    print("Vacios en prueba:", X_prueba.isna().sum()[lambda s: s > 0].to_dict())

    prep = preparacion()
    X_ent_listo = prep.fit_transform(X_ent)  # aprende solo con entrenamiento
    X_prueba_listo = prep.transform(X_prueba)  # prueba solo se transforma
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print("Vacios despues de preparar:", int(pd.isna(X_prueba_listo).sum()))
