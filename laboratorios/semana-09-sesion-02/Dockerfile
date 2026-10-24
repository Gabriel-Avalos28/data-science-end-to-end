FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

WORKDIR /app/starter

CMD ["python3", "ejercicio1_fuga.py"]
