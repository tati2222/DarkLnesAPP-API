from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(title="DarkLnes API - Versión Liviana")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check optimizado
@app.get("/")
async def root():
    return {
        "message": "DarkLnes API - Versión Liviana", 
        "status": "running",
        "version": "light-1.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API funcionando correctamente"}

# Endpoint simple para verificar funcionamiento
@app.post("/run/predict")
async def run_predict():
    return {
        "status": "ok",
        "message": "API en modo liviano - Análisis desactivado temporalmente",
        "emociones": {
            "Alegría": 0.3,
            "Neutral": 0.25,
            "Enojo": 0.15,
            "Miedo": 0.1,
            "Sorpresa": 0.1,
            "Tristeza": 0.05,
            "Disgusto": 0.05
        },
        "sd3": {
            "Maquiavelismo": 45.5,
            "Narcisismo": 52.3,
            "Psicopatía": 38.7
        }
    }

# Endpoint para análisis de video (simplificado)
@app.post("/analyze-video")
async def analyze_video(request: dict):
    return {
        "emocion_predominante": "Datos recibidos - Análisis posterior",
        "total_frames": 0,
        "duracion_video": 0,
        "emociones_detectadas": ["Datos registrados"],
        "correlaciones": {"maquiavelismo": 0, "narcisismo": 0, "psicopatia": 0},
        "frames_analizados": 0,
        "intensidad_promedio": 0,
        "variabilidad_emocional": 0,
        "aus_frecuentes": [],
        "facs_promedio": {},
        "mensaje": "Video recibido - Análisis desactivado temporalmente para optimización"
    }

# Para Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
