"""Pruebas de pipeline_proyecto.py. Se corren desde la raiz del fork con:

    pytest psets/proyecto -v

No leen la muestra de data/ ni de la Semana 9: en GitHub Actions ese archivo no
existe. Cada prueba arma sus propias ventas, escritas a mano.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import (
    CORTE,
    FUGAS,
    OBJETIVO,
    agregar_features,
    columnas_modelo,
    costo_stock,
    dividir_por_fecha,
    precio_promedio_por_serie,
    preparacion,
)


def ventas_de_prueba():
    """Cinco dias de ventas, con las columnas de la muestra: dos productos en dos tiendas,
    uno antes del lanzamiento y dos dias de 2016 (prueba)."""
    return pd.DataFrame({
        "DATE": ["2015-03-01", "2015-03-10", "2015-12-31", "2016-01-01", "2016-02-15"],
        "ITEM_ID": ["FOODS_1_001", "FOODS_1_001", "HOBBIES_1_002", "FOODS_1_001", "HOBBIES_1_002"],
        "STORE_ID": ["CA_1", "CA_1", "TX_2", "CA_1", "TX_2"],
        "CAT_ID": ["FOODS", "FOODS", "HOBBIES", "FOODS", "HOBBIES"],
        "WEEKDAY": ["Sunday", "Tuesday", "Thursday", "Friday", "Monday"],
        "SELL_PRICE": [5.0, 4.5, 10.0, 5.5, 8.0],
        "IS_WEEKEND": [True, False, False, False, False],
        "IS_EVENT_DAY": [False, False, True, True, False],
        "IS_SNAP": [False, True, False, False, True],
        "IS_PRE_LAUNCH": [False, False, False, False, False],
        "fecha_lanzamiento": ["2015-03-01", "2015-03-01", "2014-12-31", "2015-03-01", "2014-12-31"],
        "precio_promedio": [5.0, 5.0, 10.0, 5.0, 10.0],
        "UNITS": [3, 1, 0, 2, 4],
        "REVENUE": [15.0, 4.5, 0.0, 11.0, 32.0],
        "IS_SALES_SPIKE": [False, False, False, False, True],
        "IS_SUSPECTED_STOCKOUT": [False, False, True, False, False],
    })


def test_el_objetivo_y_las_fugas_no_llegan_al_modelo():
    columnas = columnas_modelo()
    assert OBJETIVO not in columnas
    assert not set(FUGAS) & set(columnas)
    assert "ITEM_ID" not in columnas

    X_ent, X_prueba, _, _ = dividir_por_fecha(agregar_features(ventas_de_prueba()))
    assert list(X_ent.columns) == columnas
    assert list(X_prueba.columns) == columnas


def test_features_calculadas_a_mano():
    datos = agregar_features(ventas_de_prueba())
    # 2015-03-10 menos 2015-03-01: 9 dias a la venta. 4.5 / 5.0 = 0.9 (10% mas barato).
    assert datos.loc[1, "dias_desde_lanzamiento"] == 9
    assert datos.loc[1, "precio_relativo"] == pytest.approx(0.9)
    # El dia del lanzamiento lleva 0 dias, y 2014-12-31 a 2015-12-31 es un anio (365 dias).
    assert datos.loc[0, "dias_desde_lanzamiento"] == 0
    assert datos.loc[2, "dias_desde_lanzamiento"] == 365
    # Las banderas pasan de True/False a 1/0.
    assert datos["IS_WEEKEND"].tolist() == [1, 0, 0, 0, 0]


def test_agregar_features_no_modifica_la_entrada():
    ventas = ventas_de_prueba()
    agregar_features(ventas)
    assert "precio_relativo" not in ventas.columns
    assert ventas["IS_WEEKEND"].dtype == bool


def test_precio_promedio_no_mira_el_futuro():
    ventas = ventas_de_prueba()
    # Un precio absurdo en 2016 y un dia antes del lanzamiento: ninguno debe entrar al promedio.
    ventas.loc[3, "SELL_PRICE"] = 1000.0
    antes = ventas.iloc[[0]].assign(DATE="2015-02-01", SELL_PRICE=500.0, IS_PRE_LAUNCH=True)
    ventas = pd.concat([ventas, antes], ignore_index=True)

    promedio = precio_promedio_por_serie(ventas, corte=CORTE)
    assert promedio.loc[("FOODS_1_001", "CA_1")] == pytest.approx((5.0 + 4.5) / 2)
    assert promedio.loc[("HOBBIES_1_002", "TX_2")] == pytest.approx(10.0)


def test_division_por_fecha_sin_dias_del_futuro_en_entrenamiento():
    datos = agregar_features(ventas_de_prueba())
    X_ent, X_prueba, y_ent, y_prueba = dividir_por_fecha(datos, corte=CORTE)
    assert datos.loc[X_ent.index, "DATE"].max() < CORTE
    assert datos.loc[X_prueba.index, "DATE"].min() >= CORTE
    assert len(X_ent) + len(X_prueba) == len(datos)
    assert y_ent.tolist() == [3, 1, 0]


def test_preparacion_acepta_categoria_nueva_y_precio_vacio():
    datos = agregar_features(ventas_de_prueba())
    X_ent, _, y_ent, _ = dividir_por_fecha(datos)
    modelo = Pipeline([("preparar", preparacion()), ("modelo", DummyRegressor(strategy="mean"))])
    modelo.fit(X_ent, y_ent)

    # Una tienda y un dia que no vio al entrenar, y un producto lanzado en 2016 sin
    # precio promedio de entrenamiento (precio_relativo vacio).
    nuevo = datos.iloc[[4]].copy()
    nuevo["STORE_ID"] = "NY_1"
    nuevo["WEEKDAY"] = "Wednesday"
    nuevo["precio_relativo"] = np.nan
    listo = modelo.named_steps["preparar"].transform(nuevo[columnas_modelo()])
    assert listo.shape[1] == modelo.named_steps["preparar"].transform(X_ent).shape[1]
    assert not np.isnan(listo).any()
    assert len(modelo.predict(nuevo[columnas_modelo()])) == 1


def test_costo_stock_castiga_mas_lo_que_falta():
    # Faltan 2 unidades (2 * 3 = 6) y sobran 2 (2 * 1 = 2): promedio (6 + 2) / 2 = 4.
    assert costo_stock([5, 1], [3, 3]) == pytest.approx(4.0)
    assert costo_stock([4, 0], [4, 0]) == 0
    # Quedarse corto por 1 cuesta mas que pasarse por 1.
    assert costo_stock([2], [1]) > costo_stock([2], [3])
