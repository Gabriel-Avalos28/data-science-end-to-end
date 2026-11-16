"""Pone el alias "produccion" en una version de "pronostico-ventas-tiendas", y lista las
versiones con sus metricas para ver la decision. Se corre desde psets/proyecto/:

    python -m servicio.promover 1        # pone produccion en la version 1
    python -m servicio.promover          # sin numero: solo lista las versiones

Despues, python servicio/exportar_modelo.py exporta la version que quedo en produccion.
Volver a la version anterior es lo mismo: promover el numero anterior y exportar otra vez.
"""

import logging
import sys

from mlflow import MlflowClient

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "pronostico-ventas-tiendas"
ALIAS = "produccion"

cliente = MlflowClient()

if len(sys.argv) > 1:
    version = sys.argv[1]
    cliente.set_registered_model_alias(NOMBRE, ALIAS, version)  #---------pone (o mueve) el alias: si estaba en otra version, se quita de ahi
    #---------la decision queda escrita en la propia version, ademas del alias
    cliente.set_model_version_tag(NOMBRE, version, "promovida_con", "servicio/promover.py")

#---------lista todas las versiones con su metrica principal, para comparar antes y despues de promover
for v in sorted(cliente.search_model_versions(f"name='{NOMBRE}'"), key=lambda v: int(v.version)):
    corrida = cliente.get_run(v.run_id)
    aliases = cliente.get_model_version(NOMBRE, v.version).aliases
    print(f"Version {v.version}: max_depth={corrida.data.params.get('max_depth')}, "
          f"min_samples_leaf={corrida.data.params.get('min_samples_leaf')}, "
          f"costo de stock {corrida.data.metrics['costo_stock_prueba']:.4f}, "
          f"MAE {corrida.data.metrics['mae_prueba']:.4f}, alias: {', '.join(aliases) or '-'}")
