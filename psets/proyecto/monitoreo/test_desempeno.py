"""Pruebas de monitoreo/desempeno.py, con predicciones y ventas escritas a mano."""

import pandas as pd
import pytest

from monitoreo.desempeno import desempeno_por_semana, indicadores_sin_verdad, unir_con_la_verdad


def predicciones_de_prueba():
    return pd.DataFrame({
        "id_prediccion": ["a", "b", "c"],
        "version_modelo": ["1", "1", "2"],
        "fecha": ["2016-05-02", "2016-05-03", "2016-05-09"],   # dos semanas: la del 2 y la del 9 de mayo
        "item_id": ["FOODS_3_090", "FOODS_3_090", "FOODS_3_090"],
        "store_id": ["CA_1", "CA_1", "CA_1"],
        "unidades_estimadas": [3.0, 3.0, 5.0],
        "serie_conocida": [True, True, False],
    })


def ventas_de_prueba():
    #---------la verdad llega por (fecha, producto, tienda); una venta de otra tienda no debe unirse
    return pd.DataFrame({
        "fecha": ["2016-05-02", "2016-05-03", "2016-05-09", "2016-05-09"],
        "item_id": ["FOODS_3_090"] * 4,
        "store_id": ["CA_1", "CA_1", "CA_1", "TX_1"],
        "unidades": [5, 1, 5, 99],
    })


def test_une_cada_prediccion_con_su_venta():
    datos = unir_con_la_verdad(predicciones_de_prueba(), ventas_de_prueba())
    assert list(datos["id_prediccion"]) == ["a", "b", "c"]
    assert list(datos["unidades"]) == [5, 1, 5]


def test_costo_por_semana_y_version():
    tabla = desempeno_por_semana(unir_con_la_verdad(predicciones_de_prueba(), ventas_de_prueba()))
    semana_2 = tabla.loc[(pd.Timestamp("2016-05-02").date(), "1")]
    # faltan 2 (2 * 3 = 6) y sobran 2 (2 * 1 = 2): costo (6 + 2) / 2 = 4, MAE (2 + 2) / 2 = 2
    assert semana_2["costo_stock"] == pytest.approx(4.0)
    assert semana_2["mae"] == pytest.approx(2.0)
    assert tabla.loc[(pd.Timestamp("2016-05-09").date(), "2"), "costo_stock"] == 0


def test_indicadores_sin_verdad():
    tabla = indicadores_sin_verdad(predicciones_de_prueba())
    assert tabla.loc[(pd.Timestamp("2016-05-02").date(), "1"), "promedio_estimado"] == 3.0
    assert tabla.loc[(pd.Timestamp("2016-05-09").date(), "2"), "series_desconocidas"] == 1.0
