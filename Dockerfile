FROM python:3.11-slim

WORKDIR /app

# Copiar la carpeta completa
COPY api_service/ ./api_service/

# Instalar requirements desde adentro de api_service
RUN pip install --no-cache-dir -r api_service/requirements.txt

CMD ["python", "api_service/api.py"]
