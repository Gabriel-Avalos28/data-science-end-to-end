"""Ejercicio 2: zona, codificacion y disponibilidad en viajes_taxi.csv.

Corre el script con:

    python3 ejercicio2_features_zona.py

Al final se imprime una comprobacion por cada TODO. No la modifiques:
cuando las cuatro digan OK, pasa a las preguntas del README.
"""

import pandas as pd

viajes = pd.read_csv("viajes_taxi.csv")
viajes = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()

ZONAS_MANHATTAN = ["Midtown", "Upper East Side", "Upper West Side", "Lower Manhattan", "Harlem"]

# TODO 1: crea la columna es_tarifa_fija (1 o 0): vale 1 cuando el viaje
# va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
# JFK Airport. Imprime cuantos viajes la tienen en 1, y la tarifa minima
# y maxima de esos viajes.
jfk_a_manhattan = (viajes["zona_origen"] == "JFK Airport") & viajes["zona_destino"].isin(ZONAS_MANHATTAN)
manhattan_a_jfk = viajes["zona_origen"].isin(ZONAS_MANHATTAN) & (viajes["zona_destino"] == "JFK Airport")
viajes["es_tarifa_fija"] = (jfk_a_manhattan | manhattan_a_jfk).astype(int) #el | es un "o", vale 1 si pasa cualquiera de los dos

fija = viajes[viajes["es_tarifa_fija"] == 1]
print("Viajes con tarifa fija:", len(fija))
print("Tarifa minima:", fija["tarifa"].min(), "Tarifa maxima:", fija["tarifa"].max())

# TODO 2: codificado, el resultado de codificar zona_origen con one-hot
# (pd.get_dummies), con el prefijo "origen" y dtype=int. Imprime los
# nombres de las columnas nuevas.
codificado = pd.get_dummies(viajes, columns=["zona_origen"], prefix="origen", dtype=int) #una columna de 1/0 por cada zona
print("Columnas one-hot:", [columna for columna in codificado.columns if columna.startswith("origen_")])

# TODO 3: codifica zona_origen de forma ordinal: un numero entero por
# zona, en orden alfabetico, en una columna origen_ordinal.
zonas_ordenadas = sorted(viajes["zona_origen"].unique()) #orden alfabetico, Brooklyn queda primero
numero_de_zona = {zona: numero for numero, zona in enumerate(zonas_ordenadas)} #a cada zona le doy un numero desde 0
viajes["origen_ordinal"] = viajes["zona_origen"].map(numero_de_zona)
print("Codigo ordinal de cada zona:", numero_de_zona)

# TODO 4: disponible_al_pedir, un diccionario con cada columna original
# del dataset y True o False segun exista en el momento en que alguien
# pide el taxi (antes del viaje). Imprime la decision de cada una.
disponible_al_pedir = {
    "id_viaje": True, #existe, pero es solo un identificador y no sirve para predecir
    "fecha_hora_inicio": True, #es el momento en que se pide el taxi
    "zona_origen": True, #el pasajero dice de donde sale
    "zona_destino": True, #y a donde va
    "distancia_km": True, #se puede estimar con la ruta antes de salir
    "duracion_min": False, #solo se sabe cuando termina el viaje
    "pasajeros": True, #se sabe cuantos se suben
    "metodo_pago": False, #se decide al pagar, al final del viaje
    "tarifa": False, #es lo que queremos predecir
    "propina": False, #se da al final del viaje
}
for columna, disponible in disponible_al_pedir.items():
    print(f"{columna}: {'existe al pedir' if disponible else 'NO existe al pedir'}")


def comprobar(numero, condicion):
    if condicion:
        print(f"TODO {numero}: OK")
    else:
        print(f"TODO {numero}: revisa")


print()
print("Comprobacion")
comprobar(1, "es_tarifa_fija" in viajes.columns and viajes["es_tarifa_fija"].sum() == 300)
comprobar(2, codificado is not None
          and len([columna for columna in codificado.columns if columna.startswith("origen_")]) == 8)
comprobar(3, "origen_ordinal" in viajes.columns
          and viajes.loc[viajes["zona_origen"] == "Brooklyn", "origen_ordinal"].iloc[0] == 0
          and viajes["origen_ordinal"].max() == 7)
comprobar(4, disponible_al_pedir is not None
          and disponible_al_pedir.get("duracion_min") is False
          and disponible_al_pedir.get("propina") is False
          and disponible_al_pedir.get("distancia_km") is True)
