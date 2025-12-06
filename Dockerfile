FROM python:3.11-slim

WORKDIR /app

# Instalamos dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos todo el proyecto
COPY . .

# Instalamos requirements desde la carpeta api_service
RUN pip install --no-cache-dir -r api_service/requirements.txt

# Exponemos puerto
ENV PORT=8000
EXPOSE 8000

# Ejecutamos la API
CMD ["python", "api_service/api.py"]
