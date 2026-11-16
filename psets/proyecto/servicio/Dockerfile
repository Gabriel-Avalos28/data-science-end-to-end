# La imagen del servicio de demanda. Se construye desde psets/proyecto/, porque necesita
# modelado/ ademas de servicio/. Antes, exporta el modelo y las referencias:
#
#     python servicio/exportar_modelo.py
#     docker build -f servicio/Dockerfile -t servicio-demanda .
#     docker run -p 8000:8000 servicio-demanda
FROM python:3.11-slim
WORKDIR /app

# Dependencias antes que codigo: si solo cambia el codigo, esta capa se reutiliza.
COPY servicio/requirements.txt servicio/requirements.txt
RUN pip install --no-cache-dir -r servicio/requirements.txt

COPY modelado/pipeline_proyecto.py modelado/
COPY servicio/api.py servicio/modelo.joblib servicio/referencias.csv servicio/modelo_info.json servicio/
EXPOSE 8000

# 0.0.0.0: escucha en todas las interfaces del contenedor, no solo en su propio localhost.
CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]
