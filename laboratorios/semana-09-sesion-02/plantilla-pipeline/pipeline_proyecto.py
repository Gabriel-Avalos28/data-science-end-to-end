import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Como corre en Docker (desde la carpeta semana-09-sesion-02):
#   docker compose run --rm -w /app/plantilla-pipeline python-app python pipeline_proyecto.py

# TODO: el archivo de tu proyecto, con las 2 features de la sesion pasada
# ya agregadas (o el script que las agrega, antes de esta linea).
# Es la muestra del 1% de obt_sales.csv.gz que guarda features_proyecto.py de la sesion 1,
# con dias_desde_lanzamiento y precio_relativo. El docker-compose la conecta en /features.
datos = pd.read_csv("/features/muestra_con_features.csv.gz")
print("Filas y columnas del archivo:", datos.shape)

OBJETIVO = "UNITS" #unidades vendidas de un producto en una tienda en un dia
NUMERICAS = ["SELL_PRICE", "precio_relativo", "dias_desde_lanzamiento"] #se imputan y se escalan
CATEGORICAS = ["CAT_ID", "STORE_ID", "WEEKDAY"] #se codifican con one-hot
YA_LISTAS = ["IS_WEEKEND", "IS_EVENT_DAY", "IS_SNAP"] #ya son 1/0, pasan sin cambios
# Fuera: REVENUE, IS_SALES_SPIKE e IS_SUSPECTED_STOCKOUT no existen al predecir (fuga),
# e ITEM_ID tiene 3049 productos, con one-hot serian 3049 columnas.

datos[YA_LISTAS] = datos[YA_LISTAS].astype(int) #vienen como True/False, las paso a 1/0

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# TODO: decide como dividir. Si tus filas tienen fecha y vas a predecir
# el futuro, divide por fecha (entrenamiento antes de una fecha de corte,
# prueba despues) en vez de al azar.
# Mi modelo predice ventas futuras, asi que divido por fecha con el mismo corte de la sesion 1.
CORTE = "2016-01-01"
es_entrenamiento = datos["DATE"] < CORTE
X_ent, X_prueba = X[es_entrenamiento], X[~es_entrenamiento]
y_ent, y_prueba = y[es_entrenamiento], y[~es_entrenamiento]
print("Entrenamiento: fechas", datos.loc[es_entrenamiento, "DATE"].min(), "a", datos.loc[es_entrenamiento, "DATE"].max())
print("Prueba: fechas", datos.loc[~es_entrenamiento, "DATE"].min(), "a", datos.loc[~es_entrenamiento, "DATE"].max())

# vacios antes de preparar: precio_relativo queda vacio para productos que salieron en 2016,
# porque no tienen precio promedio de entrenamiento
print("Vacios en entrenamiento:", X_ent.isna().sum()[lambda s: s > 0].to_dict())
print("Vacios en prueba:", X_prueba.isna().sum()[lambda s: s > 0].to_dict())

preparacion = ColumnTransformer([
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

X_ent_listo = preparacion.fit_transform(X_ent) #aprende solo con entrenamiento
X_prueba_listo = preparacion.transform(X_prueba) #prueba solo se transforma
print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
print(list(preparacion.get_feature_names_out()))

# comprobacion: las medianas del imputador tienen que ser las de entrenamiento, no las de todo
medianas = preparacion.named_transformers_["numericas"].named_steps["imputar"].statistics_
print("Medianas que aprendio el imputador:", {columna: round(float(m), 4) for columna, m in zip(NUMERICAS, medianas)})
print("Medianas de entrenamiento:", X_ent[NUMERICAS].median().round(4).to_dict())
print("Medianas de todo el dataset:", X[NUMERICAS].median().round(4).to_dict())
print("Vacios despues de preparar:", int(pd.isna(X_prueba_listo).sum()))
