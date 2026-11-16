"""Pruebas del servicio. Se corren desde la raiz del fork con:

    pytest psets/proyecto -v

TestClient manda peticiones a la API sin levantar un servidor. Las pruebas reemplazan
modelo.joblib y referencias.csv (que no existen en GitHub Actions) por un modelo y unas
referencias de prueba: revisan el contrato de la API, no la calidad del modelo.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import SERIE, columnas_modelo, preparacion
from servicio import api

PETICION = {
    "fecha": "2016-05-23",
    "item_id": "FOODS_3_090",
    "store_id": "CA_3",
    "precio": 1.0,
    "es_dia_de_evento": False,
}


def referencias_de_prueba():
    """Una sola serie conocida: FOODS_3_090 en CA_3, lanzada el 2016-05-01 a 1.25 en promedio."""
    return pd.DataFrame(
        {"fecha_lanzamiento": ["2016-05-01"], "precio_promedio": [1.25]},
        index=pd.MultiIndex.from_tuples([("FOODS_3_090", "CA_3")], names=SERIE),
    )


class ModeloEspia:
    """Un modelo que estima siempre 4 unidades y guarda lo que recibio, para revisar las
    features que calculo el servicio."""

    def __init__(self):
        self.recibido = None

    def predict(self, X):
        self.recibido = X.copy()
        return [4.0] * len(X)


def modelo_de_prueba():
    """La preparacion real con un modelo que siempre estima 3 unidades."""
    X = pd.DataFrame({
        "SELL_PRICE": [1.0, 2.0], "precio_relativo": [0.9, 1.1], "dias_desde_lanzamiento": [10, 400],
        "CAT_ID": ["FOODS", "HOBBIES"], "STORE_ID": ["CA_3", "TX_1"], "WEEKDAY": ["Monday", "Saturday"],
        "IS_WEEKEND": [0, 1], "IS_EVENT_DAY": [0, 0], "IS_SNAP": [1, 0],
    })[columnas_modelo()]
    modelo = Pipeline([("preparar", preparacion()), ("modelo", DummyRegressor(strategy="constant", constant=3.0))])
    modelo.fit(X, [3.0, 3.0])
    return modelo


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setattr(api, "modelo", modelo_de_prueba())
    monkeypatch.setattr(api, "version_modelo", "abc123")
    monkeypatch.setattr(api, "referencias", referencias_de_prueba())
    return TestClient(api.app)


def test_salud(cliente):
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "ok"
    assert respuesta.json()["modelo_cargado"] is True


def test_prediccion_valida(cliente):
    respuesta = cliente.post("/predecir", json=PETICION)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"unidades_estimadas": 3.0, "serie_conocida": True, "version_modelo": "abc123"}


def test_el_servicio_calcula_las_features_como_en_el_entrenamiento(cliente, monkeypatch):
    espia = ModeloEspia()
    monkeypatch.setattr(api, "modelo", espia)
    respuesta = cliente.post("/predecir", json=PETICION)
    assert respuesta.status_code == 200

    fila = espia.recibido.iloc[0]
    assert list(espia.recibido.columns) == columnas_modelo()
    # 2016-05-23 fue lunes, y en California el dia 23 no es dia SNAP (solo del 1 al 10).
    assert fila["WEEKDAY"] == "Monday"
    assert fila["IS_WEEKEND"] == 0
    assert fila["IS_SNAP"] == 0
    assert fila["CAT_ID"] == "FOODS"
    # 22 dias desde el lanzamiento, y 1.0 / 1.25 = 0.8 (20% mas barato que su promedio).
    assert fila["dias_desde_lanzamiento"] == 22
    assert fila["precio_relativo"] == pytest.approx(0.8)


def test_serie_desconocida_tambien_se_predice(cliente):
    peticion = {**PETICION, "item_id": "HOBBIES_9_999", "store_id": "WI_9"}
    respuesta = cliente.post("/predecir", json=peticion)
    assert respuesta.status_code == 200
    assert respuesta.json()["serie_conocida"] is False
    assert respuesta.json()["unidades_estimadas"] == 3.0


@pytest.mark.parametrize("cambio", [
    {"precio": None},            # falta el precio
    {"precio": -2.5},            # precio negativo
    {"fecha": "23 de mayo"},     # fecha mal escrita
    {"store_id": "NY_1"},        # una tienda de un estado que no existe en los datos
])
def test_entrada_invalida_recibe_422(cliente, cambio):
    peticion = {**PETICION, **cambio}
    peticion = {clave: valor for clave, valor in peticion.items() if valor is not None}
    respuesta = cliente.post("/predecir", json=peticion)
    assert respuesta.status_code == 422


def test_sin_modelo_responde_503(cliente, monkeypatch):
    monkeypatch.setattr(api, "modelo", None)
    respuesta = cliente.post("/predecir", json=PETICION)
    assert respuesta.status_code == 503
