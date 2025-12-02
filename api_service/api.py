from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn
import os
import base64
import cv2
import tempfile
from typing import Dict, List, Any
import json
import asyncio
import logging
from datetime import datetime
import uuid

# =====================================================
# CONFIGURACIÓN SUPABASE
# =====================================================
try:
    from supabase import create_client, Client
    
    # Configurar Supabase desde variables de entorno
    SUPABASE_URL = os.environ.get("https://cdhndtzuwtmvhiulvzbp.supabase.co
", "")
    SUPABASE_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNkaG5kdHp1d3RtdmhpdWx2emJwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzNTE1OTcsImV4cCI6MjA3OTkyNzU5N30.KeyAfqJuCjgSpmd0kRdjDppkJwBRlF9oGyN0ozJMt6M
", "")
    
    if SUPABASE_URL and SUPABASE_KEY:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase_available = True
        logger.info("✅ Supabase configurado correctamente")
    else:
        supabase_available = False
        logger.warning("⚠️ Supabase no configurado. Variables de entorno faltantes.")
        
except ImportError:
    logger.warning("⚠️ Biblioteca supabase no instalada. Ejecuta: pip install supabase")
    supabase_available = False
except Exception as e:
    logger.error(f"❌ Error configurando Supabase: {e}")
    supabase_available = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# MODELO DE EMOCIONES
# =====================================================
class MicroExpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.model(x)

# Cargar modelo de emociones
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Usando dispositivo: {device}")

model_emociones = MicroExpNet()
modelo_emociones_cargado = False

try:
    state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
    first_key = list(state.keys())[0]

    if first_key.startswith("model.model."):
        new_state = {k.replace("model.", "", 1): v for k, v in state.items()}
        model_emociones.load_state_dict(new_state, strict=True)
    elif first_key.startswith("model."):
        new_state = {k.replace("model.", ""): v for k, v in state.items()}
        model_emociones.model.load_state_dict(new_state, strict=True)
    else:
        model_emociones.model.load_state_dict(state, strict=True)

    model_emociones.to(device)
    model_emociones.eval()
    modelo_emociones_cargado = True
    logger.info("✅ Modelo de emociones cargado exitosamente!")
    
except Exception as e:
    logger.error(f"❌ Error cargando modelo de emociones: {e}")
    modelo_emociones_cargado = False

# =====================================================
# CONFIGURACIÓN FACS (SIMULADO)
# =====================================================
modelo_facs_cargado = False
logger.info("✅ Usando sistema FACS simulado")

# =====================================================
# FUNCIONES SUPABASE - ADAPTADAS A darklens_records
# =====================================================
async def upload_image_to_supabase(image_base64: str, analysis_id: str) -> str:
    """Sube imagen a Supabase Storage y retorna URL pública"""
    try:
        if not supabase_available:
            return None
        
        # Remover prefijo si existe
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decodificar base64
        image_bytes = base64.b64decode(image_base64)
        
        # Generar nombre de archivo único
        filename = f"{analysis_id}.jpg"
        
        # Subir a Supabase Storage
        try:
            # Intentar subir
            supabase_client.storage.from_("analisis-images").upload(
                filename,
                image_bytes,
                {"content-type": "image/jpeg"}
            )
            
            # Obtener URL pública
            image_url = supabase_client.storage.from_("analisis-images").get_public_url(filename)
            logger.info(f"✅ Imagen subida a Supabase Storage: {image_url}")
            return image_url
            
        except Exception as storage_error:
            # Si el bucket no existe, crearlo primero
            logger.warning(f"Bucket no encontrado, intentando crear: {storage_error}")
            try:
                # Nota: En Supabase, el bucket debe crearse desde el dashboard
                # Por ahora, guardamos solo la referencia
                return f"storage/analisis-images/{filename}"
            except Exception as e:
                logger.error(f"❌ Error con storage: {e}")
                return None
                
    except Exception as e:
        logger.error(f"❌ Error subiendo imagen a Supabase: {e}")
        return None

async def save_to_darklens_records(
    participant_data: Dict,
    sd3_data: Dict,
    analysis_results: Dict,
    image_base64: str = None
) -> Dict:
    """
    Guarda los resultados en la tabla darklens_records de Supabase
    
    Returns:
        Dict con status y analysis_id
    """
    try:
        # Generar ID único para el análisis
        analysis_id = str(uuid.uuid4())
        
        # Subir imagen a Supabase Storage si está disponible
        image_url = None
        if image_base64 and supabase_available:
            image_url = await upload_image_to_supabase(image_base64, analysis_id)
        
        # Preparar datos según la estructura de darklens_records
        record_data = {
            "id": analysis_id,
            "nombre": participant_data.get('nombre', 'Anónimo'),
            "edad": participant_data.get('edad', None),
            "genero": participant_data.get('genero', None),
            "pais": participant_data.get('pais', None),
            "mach": float(sd3_data.get('mach', 0.0)),
            "narc": float(sd3_data.get('narc', 0.0)),
            "psych": float(sd3_data.get('psych', 0.0)),
            "tiempo_total_seg": float(analysis_results.get('duracion_analisis', 0.0)),
            "emocion_princ": analysis_results.get('emocion_principal', 'No detectada'),
            "image_url": image_url,
            "created_at": datetime.utcnow().isoformat(),
            "total_frames": analysis_results.get('total_frames', 1),
            "duracion_video": float(analysis_results.get('duracion_analisis', 0.0)),
            "emociones_detectadas": json.dumps(analysis_results.get('emociones_detectadas', [])),
            "correlaciones": json.dumps(analysis_results.get('correlaciones', {})),
            "historia_utilizada": analysis_results.get('historia_utilizada', 'No determinada'),
            "aus_frecuentes": json.dumps(analysis_results.get('aus_frecuentes', [])),
            "facs_promedio": json.dumps(analysis_results.get('facs_promedio', {}))
        }
        
        # Guardar en Supabase si está disponible
        if supabase_available:
            try:
                response = supabase_client.table("darklens_records").insert(record_data).execute()
                
                # Verificar respuesta
                if hasattr(response, 'data') and response.data:
                    logger.info(f"✅ Registro guardado en darklens_records con ID: {analysis_id}")
                    return {
                        "status": "success",
                        "analysis_id": analysis_id,
                        "image_url": image_url,
                        "message": "Análisis guardado correctamente"
                    }
                else:
                    logger.error("❌ Error: Respuesta vacía de Supabase")
                    raise Exception("Respuesta vacía de Supabase")
                    
            except Exception as supabase_error:
                logger.error(f"❌ Error insertando en Supabase: {supabase_error}")
                # Fallback a almacenamiento local
                return await save_to_local_fallback(record_data, image_base64, analysis_id)
        else:
            # Si Supabase no está disponible, guardar localmente
            logger.warning("⚠️ Supabase no disponible, guardando localmente")
            return await save_to_local_fallback(record_data, image_base64, analysis_id)
            
    except Exception as e:
        logger.error(f"❌ Error guardando en darklens_records: {e}")
        return {
            "status": "error",
            "analysis_id": None,
            "message": f"Error guardando análisis: {str(e)}"
        }

async def save_to_local_fallback(record_data: Dict, image_base64: str, analysis_id: str) -> Dict:
    """Guarda los resultados localmente como fallback"""
    try:
        # Crear directorio si no existe
        os.makedirs("local_storage", exist_ok=True)
        
        # Guardar imagen si existe
        image_path = None
        if image_base64:
            try:
                if ',' in image_base64:
                    image_base64 = image_base64.split(',')[1]
                
                image_bytes = base64.b64decode(image_base64)
                image_path = f"local_storage/{analysis_id}.jpg"
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                    
                logger.info(f"✅ Imagen guardada localmente: {image_path}")
            except Exception as e:
                logger.error(f"❌ Error guardando imagen localmente: {e}")
        
        # Guardar datos en JSON
        record_data["image_local_path"] = image_path
        results_path = f"local_storage/{analysis_id}.json"
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Resultados guardados localmente: {results_path}")
        
        return {
            "status": "success_local",
            "analysis_id": analysis_id,
            "image_url": image_path,
            "message": "Análisis guardado localmente (Supabase no disponible)"
        }
        
    except Exception as e:
        logger.error(f"❌ Error guardando localmente: {e}")
        return {
            "status": "error",
            "analysis_id": None,
            "message": f"Error guardando localmente: {str(e)}"
        }

async def get_user_analytics(user_email: str = None, user_name: str = None) -> List[Dict]:
    """Obtiene análisis previos de un usuario desde darklens_records"""
    try:
        if not supabase_available:
            return []
        
        query = supabase_client.table("darklens_records").select("*")
        
        if user_email:
            # Si tu tabla tiene email, ajusta esto
            query = query.eq("nombre", user_name)  # Ajusta según tus campos
        elif user_name:
            query = query.eq("nombre", user_name)
        else:
            return []
        
        # Ordenar por fecha descendente
        query = query.order("created_at", desc=True).limit(10)
        response = query.execute()
        
        if hasattr(response, 'data'):
            return response.data
        elif isinstance(response, dict) and 'data' in response:
            return response['data']
        else:
            return []
            
    except Exception as e:
        logger.error(f"❌ Error obteniendo análisis del usuario: {e}")
        return []

# =====================================================
# FUNCIONES DE ANÁLISIS
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels_emociones = ["Alegría", "Tristeza", "Enojo", "Sorpresa", "Miedo", "Disgusto", "Neutral"]

def decode_image_base64(image_data: str) -> np.ndarray:
    """Decodifica una imagen en base64 a array numpy"""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
        
    except Exception as e:
        logger.error(f"Error decodificando imagen base64: {e}")
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

async def analyze_frame_emociones(frame: np.ndarray) -> Dict[str, Any]:
    """Analiza un frame usando el modelo de emociones"""
    try:
        if not modelo_emociones_cargado:
            return generate_fallback_analysis()
        
        img = Image.fromarray(frame)
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model_emociones(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        
        emotions = {labels_emociones[i]: float(probs[i]) for i in range(7)}
        emocion_principal = max(emotions, key=emotions.get)
        
        return {
            "emocion_principal": emocion_principal,
            "confianza": emotions[emocion_principal],
            "emociones": emotions,
            "facs": generate_facs_data(emocion_principal, emotions[emocion_principal]),
            "aus_detectadas": generate_aus_detected(emocion_principal),
            "modelo_utilizado": "EfficientNet-B0"
        }
        
    except Exception as e:
        logger.error(f"Error analizando frame: {e}")
        return generate_fallback_analysis()

def generate_facs_data(emocion_principal: str, confianza: float) -> List[Dict]:
    """Genera datos FACS basados en la emoción detectada (simulado)"""
    unidadesFACS = {
        "Alegría": [
            {"unidad": "AU6", "nombre": "Mejilla elevada", "intensidad": min(0.8 * confianza * 2, 0.9), "descripcion": "Contracción del músculo orbicular del ojo"},
            {"unidad": "AU12", "nombre": "Estiramiento de labios", "intensidad": min(0.9 * confianza * 2, 0.95), "descripcion": "Sonrisa genuina (Duchenne)"}
        ],
        "Tristeza": [
            {"unidad": "AU1", "nombre": "Ceja interna elevada", "intensidad": min(0.7 * confianza * 2, 0.8), "descripcion": "Expresión de preocupación"},
            {"unidad": "AU4", "nombre": "Ceja fruncida", "intensidad": min(0.6 * confianza * 2, 0.7), "descripcion": "Tensión en zona glabelar"},
            {"unidad": "AU15", "nombre": "Comisura labial hacia abajo", "intensidad": min(0.8 * confianza * 2, 0.85), "descripcion": "Expresión de desánimo"}
        ],
        "Enojo": [
            {"unidad": "AU4", "nombre": "Ceja fruncida", "intensidad": min(0.9 * confianza * 2, 0.95), "descripcion": "Tensión en entrecejo"},
            {"unidad": "AU5", "nombre": "Párpado superior elevado", "intensidad": min(0.7 * confianza * 2, 0.8), "descripcion": "Mirada intensa"},
            {"unidad": "AU7", "nombre": "Párpado inferior tensionado", "intensidad": min(0.6 * confianza * 2, 0.7), "descripcion": "Ojos entrecerrados"},
            {"unidad": "AU23", "nombre": "Labios tensionados", "intensidad": min(0.8 * confianza * 2, 0.85), "descripcion": "Boca apretada"}
        ],
        "Miedo": [
            {"unidad": "AU1", "nombre": "Ceja interna elevada", "intensidad": min(0.8 * confianza * 2, 0.85), "descripcion": "Expresión de alarma"},
            {"unidad": "AU2", "nombre": "Ceja externa elevada", "intensidad": min(0.7 * confianza * 2, 0.75), "descripcion": "Cejas arqueadas"},
            {"unidad": "AU4", "nombre": "Ceja fruncida", "intensidad": min(0.6 * confianza * 2, 0.7), "descripcion": "Preocupación"},
            {"unidad": "AU5", "nombre": "Párpado superior elevado", "intensidad": min(0.9 * confianza * 2, 0.95), "descripcion": "Ojos muy abiertos"},
            {"unidad": "AU20", "nombre": "Estiramiento horizontal de labios", "intensidad": min(0.5 * confianza * 2, 0.6), "descripcion": "Boca tensionada"}
        ],
        "Sorpresa": [
            {"unidad": "AU1", "nombre": "Ceja interna elevada", "intensidad": min(0.8 * confianza * 2, 0.85), "descripcion": "Elevación de cejas"},
            {"unidad": "AU2", "nombre": "Ceja externa elevada", "intensidad": min(0.7 * confianza * 2, 0.75), "descripcion": "Arqueo de cejas"},
            {"unidad": "AU5", "nombre": "Párpado superior elevado", "intensidad": min(0.9 * confianza * 2, 0.95), "descripcion": "Ojos abiertos"},
            {"unidad": "AU26", "nombre": "Mandíbula caída", "intensidad": min(0.6 * confianza * 2, 0.7), "descripcion": "Boca abierta"}
        ],
        "Disgusto": [
            {"unidad": "AU9", "nombre": "Nariz arrugada", "intensidad": min(0.8 * confianza * 2, 0.85), "descripcion": "Expresión de rechazo"},
            {"unidad": "AU10", "nombre": "Elevador del labio superior", "intensidad": min(0.7 * confianza * 2, 0.75), "descripcion": "Asco facial"},
            {"unidad": "AU15", "nombre": "Comisura labial hacia abajo", "intensidad": min(0.6 * confianza * 2, 0.7), "descripcion": "Desaprobación"}
        ],
        "Neutral": [
            {"unidad": "AU0", "nombre": "Expresión neutra", "intensidad": min(0.9 * confianza * 2, 0.95), "descripcion": "Sin actividad muscular significativa"}
        ]
    }
    
    return unidadesFACS.get(emocion_principal, unidadesFACS["Neutral"])

def generate_aus_detected(emocion_principal: str) -> List[str]:
    """Genera AUs detectadas basadas en la emoción"""
    aus_map = {
        "Alegría": ["AU6", "AU12"],
        "Tristeza": ["AU1", "AU4", "AU15"],
        "Enojo": ["AU4", "AU5", "AU7", "AU23"],
        "Miedo": ["AU1", "AU2", "AU4", "AU5", "AU20"],
        "Sorpresa": ["AU1", "AU2", "AU5", "AU26"],
        "Disgusto": ["AU9", "AU10", "AU15"],
        "Neutral": ["AU0"]
    }
    
    return aus_map.get(emocion_principal, ["AU0"])

def generate_fallback_analysis() -> Dict[str, Any]:
    """Genera análisis de fallback cuando el modelo falla"""
    emociones_base = {
        "Alegría": 0.3, "Neutral": 0.25, "Enojo": 0.15, 
        "Miedo": 0.1, "Sorpresa": 0.1, "Tristeza": 0.05, "Disgusto": 0.05
    }
    
    emocion_principal = max(emociones_base, key=emociones_base.get)
    
    return {
        "emocion_principal": emocion_principal,
        "confianza": 0.7,
        "emociones": emociones_base,
        "facs": generate_facs_data(emocion_principal, 0.7),
        "aus_detectadas": generate_aus_detected(emocion_principal),
        "modelo_utilizado": "Fallback"
    }

def calculate_correlations_single(resultado_frame: Dict, sd3_data: Dict) -> Dict[str, float]:
    """Calcula correlaciones entre una emoción y rasgos SD3"""
    emocion_to_trait = {
        "Alegría": "narcisismo",
        "Enojo": "maquiavelismo", 
        "Miedo": "psicopatia",
        "Neutral": "narcisismo",
        "Tristeza": "psicopatia",
        "Sorpresa": "narcisismo",
        "Disgusto": "maquiavelismo"
    }
    
    emocion = resultado_frame["emocion_principal"]
    confianza = resultado_frame["confianza"]
    
    correlaciones = {}
    for rasgo in ["maquiavelismo", "narcisismo", "psicopatia"]:
        score_sd3 = sd3_data.get(rasgo[:4], 0)  # mach, narc, psych
        
        rasgo_relacionado = emocion_to_trait.get(emocion, "narcisismo")
        
        if rasgo == rasgo_relacionado:
            correlacion = min(confianza * score_sd3 * 2, 1.0)
        else:
            correlacion = min((1 - confianza) * score_sd3 * 0.5, 0.5)
        
        correlaciones[rasgo] = float(correlacion)
    
    return correlaciones

def determinar_historia(sd3_data: Dict) -> str:
    """Determina la historia utilizada basada en SD3"""
    rasgos = {
        "maquiavelismo": sd3_data.get('mach', 0),
        "narcisismo": sd3_data.get('narc', 0),
        "psicopatia": sd3_data.get('psych', 0)
    }
    
    rasgo_predominante = max(rasgos, key=rasgos.get)
    return rasgo_predominante

# =====================================================
# FASTAPI ENDPOINTS
# =====================================================
app = FastAPI(title="DarkLnes Microexpressions API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "DarkLnes Microexpressions API", 
        "status": "running",
        "modelo_emociones": modelo_emociones_cargado,
        "supabase_available": supabase_available,
        "endpoints": {
            "/analyze-image": "Analiza una imagen y guarda en Supabase",
            "/health": "Estado del sistema",
            "/config": "Configuración actual"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "modelo_emociones": modelo_emociones_cargado,
        "supabase_available": supabase_available,
        "device": device
    }

@app.get("/config")
async def get_config():
    return {
        "supabase_url_set": bool(SUPABASE_URL),
        "supabase_key_set": bool(SUPABASE_KEY),
        "model_loaded": modelo_emociones_cargado,
        "storage_bucket": "analisis-images"
    }

@app.post("/analyze-image")
async def analyze_image(request: dict):
    try:
        logger.info("🖼️ Iniciando análisis de imagen...")
        
        # Verificar datos requeridos
        required_keys = ['image_data', 'participant_data', 'sd3_data']
        if not all(key in request for key in required_keys):
            raise HTTPException(status_code=400, detail=f"Datos incompletos. Requeridos: {required_keys}")
        
        # Decodificar imagen base64
        logger.info("📥 Decodificando imagen...")
        frame = decode_image_base64(request['image_data'])
        
        # Analizar la imagen
        logger.info("🔍 Analizando imagen...")
        analisis_emociones = await analyze_frame_emociones(frame)
        
        # Procesar resultados para una sola imagen
        emocion_principal = analisis_emociones["emocion_principal"]
        confianza = analisis_emociones["confianza"]
        
        # Calcular correlaciones con SD3
        correlaciones = calculate_correlations_single(analisis_emociones, request['sd3_data'])
        
        # Obtener AUs detectadas
        aus_frecuentes = analisis_emociones.get("aus_detectadas", [])
        
        # Calcular FACS promedio
        facs_promedio = {}
        for facs in analisis_emociones.get("facs", []):
            facs_promedio[facs["unidad"]] = facs["intensidad"]
        
        # Preparar resultados finales
        analysis_results = {
            "emocion_principal": emocion_principal,
            "confianza": confianza,
            "total_frames": 1,
            "duracion_analisis": 0.0,
            "emociones_detectadas": [emocion_principal],
            "correlaciones": correlaciones,
            "frames_analizados": 1,
            "intensidad_promedio": float(confianza),
            "variabilidad_emocional": 0.0,
            "aus_frecuentes": aus_frecuentes,
            "facs_promedio": facs_promedio,
            "historia_utilizada": determinar_historia(request['sd3_data']),
            "modelos_utilizados": {
                "emociones": analisis_emociones.get("modelo_utilizado", "Desconocido"),
                "facs": "Simulado"
            },
            "detalles_frame": analisis_emociones
        }
        
        # Guardar en Supabase
        logger.info("💾 Guardando resultados en Supabase...")
        save_result = await save_to_darklens_records(
            participant_data=request['participant_data'],
            sd3_data=request['sd3_data'],
            analysis_results=analysis_results,
            image_base64=request['image_data']
        )
        
        # Agregar metadata a la respuesta
        response_data = {
            **analysis_results,
            "participante": request['participant_data'].get('nombre', 'Anónimo'),
            "timestamp_analisis": datetime.utcnow().isoformat(),
            "tipo_analisis": "imagen",
            "storage_info": save_result
        }
        
        logger.info("✅ Análisis de imagen completado y guardado")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en análisis de imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

# Endpoint para análisis directo desde archivo
@app.post("/analyze-image-file")
async def analyze_image_file(
    file: UploadFile = File(...),
    nombre: str = "Anónimo",
    edad: int = None,
    genero: str = None,
    pais: str = None,
    mach: float = 0.0,
    narc: float = 0.0,
    psych: float = 0.0
):
    try:
        logger.info(f"🖼️ Analizando imagen de {nombre}...")
        
        # Leer archivo y convertir a base64
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        image_data = f"data:{file.content_type};base64,{image_base64}"
        
        # Preparar datos
        participant_data = {
            "nombre": nombre,
            "edad": edad,
            "genero": genero,
            "pais": pais
        }
        
        sd3_data = {
            "mach": mach,
            "narc": narc,
            "psych": psych
        }
        
        # Crear request
        request = {
            "image_data": image_data,
            "participant_data": participant_data,
            "sd3_data": sd3_data
        }
        
        # Llamar al endpoint principal
        return await analyze_image(request)
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de archivo: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

# Endpoint para obtener análisis previos
@app.get("/user-analytics/{user_name}")
async def get_user_analytics_endpoint(user_name: str):
    try:
        analytics = await get_user_analytics(user_name=user_name)
        return {
            "user": user_name,
            "total_analyses": len(analytics),
            "analytics": analytics[:10]  # Limitar a 10 resultados
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo analytics: {str(e)}")

# Para ejecutar en Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
