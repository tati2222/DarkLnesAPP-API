# Usa una imagen base ligera con Python
FROM python:3.10-slim

# Evitar prompts
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY api_service/requirements.txt /app/requirements.txt

# Instalar dependencias Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el código completo
COPY . /app

# Exponer el puerto donde Uvicorn escucha
EXPOSE 8000

# Comando de inicio en Railway
CMD ["uvicorn", "api_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
