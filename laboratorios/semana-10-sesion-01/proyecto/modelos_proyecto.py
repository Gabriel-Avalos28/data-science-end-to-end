import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Como corre en Docker (desde la carpeta semana-10-sesion-01):
#   docker compose run --rm -w /app/proyecto python-app python modelos_proyecto.py

# La misma muestra del 1% de obt_sales.csv.gz con mis 2 features de la Semana 9, Sesion 1.
# El docker-compose la conecta en /features.
datos = pd.read_csv("/features/muestra_con_features.csv.gz")
print("Filas y columnas del archivo:", datos.shape)

# Las mismas columnas del pipeline de la Semana 9, Sesion 2
OBJETIVO = "UNITS" #unidades vendidas de un producto en una tienda en un dia
NUMERICAS = ["SELL_PRICE", "precio_relativo", "dias_desde_lanzamiento"] #se imputan y se escalan
CATEGORICAS = ["CAT_ID", "STORE_ID", "WEEKDAY"] #se codifican con one-hot
YA_LISTAS = ["IS_WEEKEND", "IS_EVENT_DAY", "IS_SNAP"] #ya son 1/0, pasan sin cambios

datos[YA_LISTAS] = datos[YA_LISTAS].astype(int) #vienen como True/False, las paso a 1/0

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# La misma division por fecha: entreno hasta 2015 y pruebo con 2016
CORTE = "2016-01-01"
es_entrenamiento = datos["DATE"] < CORTE
X_ent, X_prueba = X[es_entrenamiento], X[~es_entrenamiento]
y_ent, y_prueba = y[es_entrenamiento], y[~es_entrenamiento]
print("Entrenamiento:", X_ent.shape, "| Prueba:", X_prueba.shape)
print("Mediana de UNITS en entrenamiento:", y_ent.median(), "| promedio:", round(y_ent.mean(), 4))
print("Dias con 0 unidades en entrenamiento:", f"{(y_ent == 0).mean():.1%}")


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


resultados = []


# Igual que la funcion evaluar de los ejercicios: entrena el pipeline completo,
# mide el error absoluto medio en entrenamiento y en prueba, y lo anota en la tabla
def evaluar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    resultados.append({
        "modelo": nombre,
        "mae_entrenamiento": round(mae_ent, 4),
        "mae_prueba": round(mae_prueba, 4),
        "brecha": round(mae_prueba - mae_ent, 4),
        "segundos": round(segundos, 3),
    })
    print(f"{nombre:34s} entrenamiento {mae_ent:7.4f}   prueba {mae_prueba:7.4f}   {segundos:7.1f} s")
    return pipeline


# Linea base: siempre la mediana de entrenamiento. Mi metrica es MAE y la constante
# que menos MAE da es la mediana. No uso "manana igual que hoy" porque la muestra es
# un 1% al azar: casi nunca esta el dia anterior del mismo producto en la misma tienda.
evaluar("linea base (mediana)", DummyRegressor(strategy="median"))
# como 60% de los dias tienen 0 unidades, la mediana es 0; comparo tambien con el promedio
evaluar("linea base (promedio)", DummyRegressor(strategy="mean"))

# Modelo lineal: primero sin regularizar y despues con Ridge, para ver si hace falta
evaluar("lineal", LinearRegression())
evaluar("ridge alpha=10", Ridge(alpha=10.0))

# Un arbol sin limite, para ver como se ve el sobreajuste en mis datos
evaluar("arbol sin limite", DecisionTreeRegressor(random_state=42))

# Ensamblajes. Con 425 mil filas, el bosque va regularizado (profundidad y hojas
# minimas) para que no memorice y para que no tarde demasiado.
evaluar("bosque aleatorio", RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=50,
                                                  n_jobs=-1, random_state=42))
evaluar("gradient boosting", GradientBoostingRegressor(random_state=42))
# El mismo boosting pero optimizando el error absoluto, que es mi metrica
evaluar("gradient boosting (absoluto)", GradientBoostingRegressor(loss="absolute_error", random_state=42))

tabla = pd.DataFrame(resultados)
tabla.to_csv("resultados_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))

base = tabla.loc[tabla["modelo"] == "linea base (mediana)", "mae_prueba"].iloc[0]
mejor = tabla.loc[tabla["mae_prueba"].idxmin()]
print()
print("Mejor modelo en prueba:", mejor["modelo"], "con MAE", mejor["mae_prueba"])
print(f"Mejora sobre la linea base: {base - mejor['mae_prueba']:.4f} unidades ({1 - mejor['mae_prueba'] / base:.1%})")
