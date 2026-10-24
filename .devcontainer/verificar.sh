#!/usr/bin/env bash
# Verifica que cada herramienta del curso responde. Corre al crear el Codespace y
# puedes repetirla cuando quieras: bash .devcontainer/verificar.sh
set -euo pipefail

echo "--- Verificacion del ambiente ---"
python --version
java -version 2>&1 | head -n 1
terraform -version | head -n 1
docker --version
docker compose version
dbt --version | head -n 3
python - <<'PY'
import duckdb, kafka, matplotlib, pandas, pyspark, sklearn
print(f"pandas {pandas.__version__} | duckdb {duckdb.__version__} | pyspark {pyspark.__version__}")
print(f"scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__} | kafka-python {kafka.__version__}")
PY

echo
echo "Ambiente listo. Si algo de lo anterior fallo, avisame antes de la proxima sesion."
