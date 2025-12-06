import os
import io
import json
from datetime import datetime
import requests
import logging

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

import pandas as pd
from scipy import stats

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from supabase import create_client

# ========================================
# CONFIGURACIÓN
# ========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase
SUPABASE_URL = "https://cdhndtzuwtmvhiulvzbp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNkaG5kdHp1d3RtdmhpdWx2emJwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NDM1MTU5NywiZXhwIjoyMDc5OTI3NTk3fQ.-vqSP3Vy1qLPoDcTZfo58lhcs1ydTgsgPVh8yGyX5eU"

# URL del modelo en Supabase Storage
MODEL_URL = "https://cdhndtzuwtmvhiulvzbp.supabase.co/storage/v1/object/public/modelos/modelo_microexpresiones.pth"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_microexpresiones.pth")

# Inicializar Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Inicializar FastAPI
app = FastAPI(title="DarkLens API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Variables globales
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# FUNCIÓN FACS SIMULADA
# ========================================
def generar_facs_simulada(emocion, confianza):
    """
    Genera datos FACS simulados basados en la emoción detectada.
    Esto permite que el frontend muestre datos FACS aunque no tengamos MediaPipe.
    """
    
    # Base de datos de Action Units por emoción
    emocion_a_aus = {
        # Felicidad / Alegría
        "happiness": [
            {"code": "AU06", "name": "Cheek Raiser", "base_intensity": 0.8},
            {"code": "AU12", "name": "Lip Corner Puller", "base_intensity": 0.9},
            {"code": "AU25", "name": "Lips Part", "base_intensity": 0.4}
        ],
        "felicidad": [
            {"code": "AU06", "name": "Cheek Raiser", "base_intensity": 0.8},
            {"code": "AU12", "name": "Lip Corner Puller", "base_intensity": 0.9},
            {"code": "AU25", "name": "Lips Part", "base_intensity": 0.4}
        ],
        "alegría": [
            {"code": "AU06", "name": "Cheek Raiser", "base_intensity": 0.8},
            {"code": "AU12", "name": "Lip Corner Puller", "base_intensity": 0.9},
            {"code": "AU25", "name": "Lips Part", "base_intensity": 0.4}
        ],
        
        # Tristeza
        "sadness": [
            {"code": "AU01", "name": "Inner Brow Raiser", "base_intensity": 0.7},
            {"code": "AU04", "name": "Brow Lowerer", "base_intensity": 0.6},
            {"code": "AU15", "name": "Lip Corner Depressor", "base_intensity": 0.8}
        ],
        "tristeza": [
            {"code": "AU01", "name": "Inner Brow Raiser", "base_intensity": 0.7},
            {"code": "AU04", "name": "Brow Lowerer", "base_intensity": 0.6},
            {"code": "AU15", "name": "Lip Corner Depressor", "base_intensity": 0.8}
        ],
        
        # Enojo / Ira
        "anger": [
            {"code": "AU04", "name": "Brow Lowerer", "base_intensity": 0.9},
            {"code": "AU05", "name": "Upper Lid Raiser", "base_intensity": 0.7},
            {"code": "AU07", "name": "Lid Tightener", "base_intensity": 0.6},
            {"code": "AU23", "name": "Lip Tightener", "base_intensity": 0.5}
        ],
        "enojo": [
            {"code": "AU04", "name": "Brow Lowerer", "base_intensity": 0.9},
            {"code": "AU05", "name": "Upper Lid Raiser", "base_intensity": 0.7},
            {"code": "AU07", "name": "Lid Tightener", "base_intensity": 0.6},
            {"code": "AU23", "name": "Lip Tightener", "base_intensity": 0.5}
        ],
        "ira": [
            {"code": "AU04", "name": "Brow Lowerer", "base_intensity": 0.9},
            {"code": "AU05", "name": "Upper Lid Raiser", "base_intensity": 0.7},
            {"code": "AU07", "name": "Lid Tightener", "base_intensity": 0.6},
            {"code": "AU23", "name": "Lip Tightener", "base_intensity": 0.5}
        ],
        
        # Miedo
        "fear": [
            {"code": "AU01", "name": "Inner Brow Raiser", "base_intensity": 0.8},
            {"code": "AU02", "name": "Outer Brow Raiser", "base_intensity": 0.7},
            {"code": "AU05", "name": "Upper Lid Raiser", "base_intensity": 0.9},
            {"code": "AU20", "name": "Lip Stretcher", "base_intensity": 0.6},
            {"code": "AU25", "name": "Lips Part", "base_intensity": 0.5}
        ],
        "miedo": [
            {"code": "AU01", "name": "Inner Brow Raiser", "base_intensity": 0.8},
            {"code": "AU02", "name": "Outer Brow Raiser", "base_intensity": 0.7},
            {"code": "AU05", "name": "Upper Lid Raiser", "base_intensity": 0.9},
            {"code": "AU20", "name": "Lip Stretcher", "base_intensity": 0.6},
            {"code": "AU25", "name": "Lips Part", "base_intensity": 0.5}
        ],
        
        # Sorpresa
        "surprise": [
            {"code": "AU01", "name": "Inner Brow Raiser", "base_intensity": 0.9},
            {"code": "AU02", "name": "Outer Brow Raiser", "base_intensity": 0.9},
            {"code": "AU05", "name": "Upper Lid Raiser", "base_intensity": 0.8},
            {"code": "AU25", "name": "Lips Part", "base_intensity": 0.7},
            {"code": "AU26", "name": "Jaw Drop", "base_intensity": 0.6}
        ],
        "sorpresa": [
            {"code": "AU01", "name": "Inner Brow Raiser", "base_intensity": 0.9},
            {"code": "AU02", "name": "Outer Brow Raiser", "base_intensity": 0.9},
            {"code": "AU05", "name": "Upper Lid Raiser", "base_intensity": 0.8},
            {"code": "AU25", "name": "Lips Part", "base_intensity": 0.7},
            {"code": "AU26", "name": "Jaw Drop", "base_intensity": 0.6}
        ],
        
        # Asco
        "disgust": [
            {"code": "AU09", "name": "Nose Wrinkler", "base_intensity": 0.8},
            {"code": "AU10", "name": "Upper Lip Raiser", "base_intensity": 0.7},
            {"code": "AU17", "name": "Chin Raiser", "base_intensity": 0.5}
        ],
        "asco": [
            {"code": "AU09", "name": "Nose Wrinkler", "base_intensity": 0.8},
            {"code": "AU10", "name": "Upper Lip Raiser", "base_intensity": 0.7},
            {"code": "AU17", "name": "Chin Raiser", "base_intensity": 0.5}
        ],
        
        # Neutral
        "neutral": [
            {"code": "AU25", "name": "Lips Part", "base_intensity": 0.3},
            {"code": "AU43", "name": "Eyes Closed", "base_intensity": 0.2}
        ],
        "neutro": [
            {"code": "AU25", "name": "Lips Part", "base_intensity": 0.3},
            {"code": "AU43", "name": "Eyes Closed", "base_intensity": 0.2}
        ]
    }
    
    # Normalizar nombre de emoción
    emocion_lower = emocion.lower()
    
    # Obtener AUs para la emoción o usar neutral por defecto
    aus_base = emocion_a_aus.get(emocion_lower, emocion_a_aus["neutral"])
    
    # Generar AUs con intensidades ajustadas por confianza
    action_units = []
    for au in aus_base:
        # Ajustar intensidad basada en confianza
        intensity = au["base_intensity"] * confianza
        
        # Añadir pequeña variación para hacerlo más realista
        import random
        intensity_variada = max(0.1, min(0.99, intensity + random.uniform(-0.1, 0.1)))
        
        action_units.append({
            "code": au["code"],
            "au": au["code"],
            "numero": int(au["code"][2:]) if au["code"][2:].isdigit() else 0,
            "name": au["name"],
            "intensity": round(intensity_variada, 2),
            "description": f"{au['name']} - expresión de {emocion}"
        })
    
    # Generar interpretación
    interpretation = {
        "primary_emotion": emocion,
        "confidence": round(confianza, 2),
        "microexpression_indicators": [
            {
                "type": f"Expresión de {emocion}",
                "authenticity": "Alta" if confianza > 0.7 else "Media" if confianza > 0.4 else "Baja",
                "note": f"Basado en análisis de emociones con {confianza*100:.0f}% de confianza"
            }
        ],
        "authenticity_score": round(confianza * 0.8, 2),
        "notes": [f"Análisis FACS generado automáticamente para {emocion}"]
    }
    
    return {
        "action_units": action_units,
        "interpretation": interpretation,
        "confidence": round(confianza, 2),
        "face_detected": True
    }

# ========================================
# FUNCIONES DE UTILIDAD
# ========================================
def download_model():
    """Descargar modelo desde Supabase si no existe"""
    if os.path.exists(MODEL_PATH):
        logger.info(f"Modelo ya existe: {MODEL_PATH}")
        return True
    
    try:
        logger.info(f"Descargando modelo desde: {MODEL_URL}")
        response = requests.get(MODEL_URL, timeout=60)
        
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            logger.info(f"Modelo descargado: {os.path.getsize(MODEL_PATH)} bytes")
            return True
        else:
            logger.error(f"Error descargando modelo: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False

def load_model():
    """Cargar modelo PyTorch"""
    global model
    
    if not os.path.exists(MODEL_PATH):
        logger.error("Modelo no encontrado")
        return False
    
    try:
        # Crear modelo EfficientNet-B0
        model_loaded = models.efficientnet_b0(weights=None)
        num_classes = 7
        in_features = model_loaded.classifier[1].in_features
        model_loaded.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Cargar pesos
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        if isinstance(checkpoint, dict):
            model_loaded.load_state_dict(checkpoint)
        else:
            model_loaded = checkpoint
        
        model_loaded.to(device)
        model_loaded.eval()
        
        model = model_loaded
        logger.info("✅ Modelo cargado correctamente")
        return True
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return False

# ========================================
# CONFIGURACIÓN DEL MODELO
# ========================================
EMOTION_LABELS = {
    0: "anger",
    1: "disgust", 
    2: "fear",
    3: "happiness",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

# Mapeo a español para consistencia
EMOTION_TRANSLATIONS = {
    "anger": "enojo",
    "disgust": "asco", 
    "fear": "miedo",
    "happiness": "felicidad",
    "neutral": "neutral",
    "sadness": "tristeza",
    "surprise": "sorpresa"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========================================
# ENDPOINTS
# ========================================
@app.get("/")
async def root():
    return {
        "message": "DarkLens API",
        "version": "1.0",
        "status": "online",
        "model_loaded": model is not None,
        "facs_mode": "simulated",
        "endpoints": {
            "/": "GET - Información de la API",
            "/health": "GET - Estado del sistema",
            "/analyze": "POST - Analizar imagen",
            "/test": "GET - Prueba de conexión"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/test")
async def test_endpoint():
    return {
        "message": "API funcionando correctamente",
        "facs_simulation": "active"
    }

# ========================================
# ENDPOINT PRINCIPAL - ANALYZE
# ========================================
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    nombre: str = Form(""),
    edad: int = Form(0),
    genero: str = Form(""),
    pais: str = Form(""),
    mach: float = Form(0.0),
    narc: float = Form(0.0),
    psych: float = Form(0.0),
    tiempo_total_seg: float = Form(0.0),
    historia_utilizada: str = Form(""),
    tipo_captura: str = Form("imagen"),
    include_facs: bool = Form(True),
):
    """
    Endpoint principal para analizar imágenes
    """
    # Verificar modelo
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    logger.info(f"Iniciando análisis para: {nombre}")
    
    # 1. Procesar imagen
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"Imagen cargada: {img.size}")
    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error en imagen: {str(e)}")
    
    # 2. Predecir emoción
    try:
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
            # Obtener resultados
            pred_idx = int(torch.argmax(probs))
            emotion_english = EMOTION_LABELS.get(pred_idx, "unknown")
            emotion_spanish = EMOTION_TRANSLATIONS.get(emotion_english, emotion_english)
            confidence = float(probs[pred_idx])
            
            # Todas las probabilidades
            all_probs = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
        
        logger.info(f"Emoción detectada: {emotion_spanish} ({confidence:.2%})")
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
    
    # 3. Generar FACS simulada
    facs_data = None
    if include_facs:
        try:
            facs_data = generar_facs_simulada(emotion_spanish, confidence)
            logger.info(f"FACS simulada generada: {len(facs_data['action_units'])} AUs")
        except Exception as e:
            logger.error(f"Error generando FACS: {str(e)}")
            facs_data = {
                "action_units": [],
                "interpretation": {
                    "primary_emotion": emotion_spanish,
                    "confidence": confidence,
                    "microexpression_indicators": [],
                    "authenticity_score": 0.5,
                    "notes": ["Error generando análisis FACS"]
                }
            }
    
    # 4. Subir imagen a Supabase Storage (opcional)
    image_url = ""
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{nombre}_{timestamp}.jpg" if nombre else f"img_{timestamp}.jpg"
        storage_path = f"microexpresiones/{filename}"
        
        # Convertir imagen a bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_bytes = img_byte_arr.getvalue()
        
        # Subir a Supabase
        upload_result = supabase.storage.from_("DARKLENS-IMAGES").upload(
            storage_path,
            img_bytes,
            {"content-type": "image/jpeg"}
        )
        
        # Obtener URL pública si la subida fue exitosa
        if not hasattr(upload_result, 'error') or not upload_result.error:
            url_response = supabase.storage.from_("DARKLENS-IMAGES").get_public_url(storage_path)
            if hasattr(url_response, 'public_url'):
                image_url = url_response.public_url
            elif isinstance(url_response, str):
                image_url = url_response
            
            logger.info(f"Imagen subida a Supabase: {image_url}")
    except Exception as e:
        logger.warning(f"No se pudo subir imagen: {str(e)}")
    
    # 5. Preparar datos para la base de datos
    record_data = {
        "nombre": nombre or "Anónimo",
        "edad": int(edad) if edad else 0,
        "genero": genero or "",
        "pais": pais or "",
        "mach": float(mach),
        "narc": float(narc),
        "psych": float(psych),
        "tiempo_total_seg": float(tiempo_total_seg),
        "emocion_principal": emotion_spanish,
        "confianza_emocion": confidence,
        "historia_utilizada": historia_utilizada or "",
        "tipo_captura": tipo_captura or "imagen",
        "imagen_url": image_url,
        "imagen_analizada": True,
        "include_facs": include_facs
    }
    
    # Agregar datos FACS específicos si están disponibles
    if facs_data and include_facs:
        try:
            # Extraer códigos de AUs frecuentes
            aus_frecuentes = [au["code"] for au in facs_data.get("action_units", [])]
            if aus_frecuentes:
                record_data["aus_frecuentes"] = aus_frecuentes
            
            # Guardar análisis FACS completo como JSON
            facs_json = json.dumps(facs_data, ensure_ascii=False)
            record_data["facs_json"] = facs_json
            
            # También guardar en análisis_completo
            analisis_completo = {
                "timestamp": datetime.utcnow().isoformat(),
                "emotion_detection": {
                    "primary": emotion_spanish,
                    "confidence": confidence,
                    "all_probabilities": all_probs
                },
                "facs_analysis": facs_data
            }
            record_data["analisis_completo"] = json.dumps(analisis_completo, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"Error procesando datos FACS para BD: {str(e)}")
    else:
        # Si no hay FACS, guardar análisis básico
        analisis_completo = {
            "timestamp": datetime.utcnow().isoformat(),
            "emotion_detection": {
                "primary": emotion_spanish,
                "confidence": confidence,
                "all_probabilities": all_probs
            }
        }
        record_data["analisis_completo"] = json.dumps(analisis_completo, ensure_ascii=False)
    
    # 6. Guardar en base de datos Supabase
    try:
        db_response = supabase.table("darklens_records").insert(record_data).execute()
        
        if hasattr(db_response, 'error') and db_response.error:
            logger.warning(f"Error insertando en BD: {db_response.error}")
        else:
            logger.info("Registro guardado en Supabase")
    except Exception as e:
        logger.warning(f"Excepción al guardar en BD: {str(e)}")
    
    # 7. Preparar respuesta para el frontend
    response_data = {
        "success": True,
        "message": "Análisis completado exitosamente",
        "timestamp": datetime.utcnow().isoformat(),
        
        # Datos de emoción
        "emocion_detectada": emotion_spanish,
        "emocion_principal": emotion_spanish,
        "confianza": round(confidence, 4),
        "probabilidades": all_probs,
        
        # Datos de imagen
        "imagen_url": image_url,
        "imagen_procesada": True,
        
        # Datos FACS (SIEMPRE INCLUIDOS si se solicitó)
        "facs": facs_data if include_facs else {
            "action_units": [],
            "interpretation": {
                "primary_emotion": emotion_spanish,
                "confidence": confidence,
                "microexpression_indicators": [],
                "authenticity_score": 0.5,
                "notes": ["FACS no solicitado en este análisis"]
            }
        },
        
        # Metadatos
        "modelo_utilizado": "EfficientNet-B0",
        "facs_generado": "simulado" if include_facs else "no"
    }
    
    # Agregar estadísticas FACS si están disponibles
    if facs_data and "action_units" in facs_data and include_facs:
        aus = facs_data["action_units"]
        if aus:
            response_data["facs_stats"] = {
                "total_aus": len(aus),
                "aus_codes": [au["code"] for au in aus],
                "aus_intensities": {au["code"]: au["intensity"] for au in aus},
                "average_intensity": sum(au["intensity"] for au in aus) / len(aus)
            }
    
    logger.info(f"Análisis completado para {nombre}")
    
    return JSONResponse(content=response_data)

# ========================================
# INICIALIZACIÓN
# ========================================
@app.on_event("startup")
async def startup():
    """Inicializar la aplicación"""
    logger.info("🚀 Iniciando DarkLens API...")
    
    # 1. Descargar modelo si no existe
    if not os.path.exists(MODEL_PATH):
        logger.info("Descargando modelo desde Supabase...")
        download_success = download_model()
        if not download_success:
            logger.error("No se pudo descargar el modelo")
    
    # 2. Cargar modelo
    load_model()
    
    # 3. Mostrar estado
    logger.info("=" * 50)
    logger.info("CONFIGURACIÓN INICIALIZADA:")
    logger.info(f"Modelo cargado: {model is not None}")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"FACS: Modo simulado activado")
    logger.info("=" * 50)

# ========================================
# EJECUCIÓN
# ========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
