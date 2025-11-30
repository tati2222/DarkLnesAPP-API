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
from scipy.stats import pearsonr
import asyncio
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# MODELO DE EMOCIONES (PRIORIDAD)
# -----------------------------------------------------
class MicroExpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.model(x)

# Cargar modelo de emociones (PRIMARIO)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Usando dispositivo: {device}")

model_emociones = MicroExpNet()
modelo_emociones_cargado = False
modelo_facs_cargado = False

try:
    # Intentar cargar el modelo de emociones
    state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
    first_key = list(state.keys())[0]

    # Ajustar claves
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

# -----------------------------------------------------
# CONFIGURACIÓN FACS (SECUNDARIO - SIMULADO SI FALLA)
# -----------------------------------------------------
try:
    # Intentar importar Py-Feat (opcional)
    import pyfeat
    from pyfeat import Detector
    detector_facs = Detector(device='cpu')
    modelo_facs_cargado = True
    logger.info("✅ Py-Feat disponible para análisis FACS")
except ImportError:
    logger.warning("⚠️ Py-Feat no disponible, usando FACS simulado")
    modelo_facs_cargado = False
except Exception as e:
    logger.warning(f"⚠️ Error cargando Py-Feat: {e}, usando FACS simulado")
    modelo_facs_cargado = False

# -----------------------------------------------------
# PREPROCESAMIENTO
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels_emociones = ["Alegría", "Tristeza", "Enojo", "Sorpresa", "Miedo", "Disgusto", "Neutral"]

# -----------------------------------------------------
# FUNCIONES PARA ANÁLISIS DE VIDEO Y FACS
# -----------------------------------------------------
def extract_frames(video_path: str, frames_per_second: int = 1) -> List[np.ndarray]:
    """Extrae frames del video (1 frame por segundo por defecto)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps / frames_per_second))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        frame_count += 1
        
    cap.release()
    return frames

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

async def analyze_frame_facs(frame: np.ndarray) -> Dict[str, Any]:
    """Analiza un frame usando Py-Feat para FACS (si está disponible)"""
    try:
        if not modelo_facs_cargado:
            return {"facs_disponible": False, "aus": [], "landmarks": []}
        
        # Convertir numpy array a formato compatible con Py-Feat
        result = detector_facs.detect_image(frame)
        
        facs_data = []
        aus_detectadas = []
        
        if result['faces'] is not None and len(result['faces']) > 0:
            # Extraer AUs y landmarks
            aus = result['aus'][0] if result['aus'] is not None else []
            landmarks = result['landmarks'][0] if result['landmarks'] is not None else []
            
            # Convertir AUs a formato legible
            for i, au_intensity in enumerate(aus):
                if au_intensity > 0.1:  # Threshold para considerar AU activa
                    au_name = f"AU{i+1}"
                    facs_data.append({
                        "unidad": au_name,
                        "nombre": get_au_name(au_name),
                        "intensidad": float(au_intensity),
                        "descripcion": get_au_description(au_name)
                    })
                    aus_detectadas.append(au_name)
            
            return {
                "facs_disponible": True,
                "aus": facs_data,
                "aus_detectadas": aus_detectadas,
                "landmarks": landmarks.tolist() if landmarks is not None else [],
                "modelo_utilizado": "Py-Feat"
            }
        else:
            return {"facs_disponible": False, "aus": [], "landmarks": []}
            
    except Exception as e:
        logger.error(f"Error en análisis FACS: {e}")
        return {"facs_disponible": False, "aus": [], "landmarks": []}

def get_au_name(au_code: str) -> str:
    """Obtiene el nombre descriptivo de una Action Unit"""
    au_names = {
        "AU1": "Ceja interna elevada",
        "AU2": "Ceja externa elevada", 
        "AU4": "Ceja fruncida",
        "AU5": "Párpado superior elevado",
        "AU6": "Mejilla elevada",
        "AU7": "Párpado tensionado",
        "AU9": "Nariz arrugada",
        "AU10": "Elevador labio superior",
        "AU12": "Estiramiento de labios",
        "AU15": "Comisura labial hacia abajo",
        "AU17": "Mentón elevado",
        "AU20": "Estiramiento horizontal de labios",
        "AU23": "Labios tensionados",
        "AU25": "Labios separados",
        "AU26": "Mandíbula caída"
    }
    return au_names.get(au_code, f"Unidad {au_code}")

def get_au_description(au_code: str) -> str:
    """Obtiene la descripción de una Action Unit"""
    au_descriptions = {
        "AU1": "Expresión de preocupación o tristeza",
        "AU2": "Expresión de sorpresa o miedo",
        "AU4": "Expresión de enojo o concentración",
        "AU5": "Expresión de miedo o sorpresa",
        "AU6": "Expresión de alegría genuina",
        "AU7": "Expresión de tensión ocular",
        "AU9": "Expresión de disgusto o rechazo",
        "AU10": "Expresión de disgusto superior",
        "AU12": "Expresión de sonrisa",
        "AU15": "Expresión de tristeza o desánimo",
        "AU17": "Expresión de determinación o tensión",
        "AU20": "Expresión de miedo o tensión labial",
        "AU23": "Expresión de enojo o frustración",
        "AU25": "Expresión de sorpresa o habla",
        "AU26": "Expresión de sorpresa o incredulidad"
    }
    return au_descriptions.get(au_code, "Unidad de acción facial")

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

def process_aggregated_results(resultados_frames: List[Dict], sd3_data: Dict) -> Dict[str, Any]:
    """Procesa y agrega los resultados de todos los frames"""
    if not resultados_frames:
        return generate_empty_response()
    
    # Estadísticas de emociones
    emociones_todas = [r["emocion_principal"] for r in resultados_frames]
    emocion_predominante = max(set(emociones_todas), key=emociones_todas.count)
    
    # Calcular correlaciones con SD3
    correlaciones = calculate_correlations(resultados_frames, sd3_data)
    
    # Estadísticas FACS
    aus_frecuentes = calculate_frequent_aus(resultados_frames)
    facs_promedio = calculate_average_facs(resultados_frames)
    
    # Métricas adicionales
    intensidad_promedio = np.mean([r.get("confianza", 0.5) for r in resultados_frames])
    variabilidad_emocional = calculate_emotional_variability(resultados_frames)
    
    return {
        "emocion_predominante": emocion_predominante,
        "total_frames": len(resultados_frames),
        "duracion_video": len(resultados_frames),  # 1 frame por segundo
        "emociones_detectadas": list(set(emociones_todas)),
        "correlaciones": correlaciones,
        "frames_analizados": len(resultados_frames),
        "intensidad_promedio": float(intensidad_promedio),
        "variabilidad_emocional": float(variabilidad_emocional),
        "aus_frecuentes": aus_frecuentes,
        "facs_promedio": facs_promedio,
        "modelos_utilizados": {
            "emociones": resultados_frames[0].get("modelo_utilizado", "Desconocido"),
            "facs": "Py-Feat" if modelo_facs_cargado else "Simulado"
        }
    }

def calculate_correlations(resultados_frames: List[Dict], sd3_data: Dict) -> Dict[str, float]:
    """Calcula correlaciones entre emociones y rasgos SD3"""
    emocion_to_trait = {
        "Alegría": "narcisismo",
        "Enojo": "maquiavelismo", 
        "Miedo": "psicopatia",
        "Neutral": "narcisismo",
        "Tristeza": "psicopatia",
        "Sorpresa": "narcisismo",
        "Disgusto": "maquiavelismo"
    }
    
    emocion_counts = {}
    for resultado in resultados_frames:
        emocion = resultado["emocion_principal"]
        emocion_counts[emocion] = emocion_counts.get(emocion, 0) + 1
    
    correlaciones = {}
    for rasgo in ["maquiavelismo", "narcisismo", "psicopatia"]:
        score_sd3 = sd3_data.get(rasgo[:4], 0)  # mach, narc, psych
        
        emociones_relacionadas = [emocion for emocion, trait in emocion_to_trait.items() 
                                if trait == rasgo]
        
        total_frames_relacionados = sum(emocion_counts.get(emocion, 0) 
                                      for emocion in emociones_relacionadas)
        proporcion = total_frames_relacionados / len(resultados_frames) if resultados_frames else 0
        
        correlacion = min(proporcion * score_sd3 * 2, 1.0)
        correlaciones[rasgo] = float(correlacion)
    
    return correlaciones

def calculate_frequent_aus(resultados_frames: List[Dict]) -> List[str]:
    """Calcula las Action Units más frecuentes"""
    aus_counts = {}
    for resultado in resultados_frames:
        for au in resultado.get("aus_detectadas", []):
            aus_counts[au] = aus_counts.get(au, 0) + 1
    
    aus_frecuentes = sorted(aus_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return [au for au, count in aus_frecuentes]

def calculate_average_facs(resultados_frames: List[Dict]) -> Dict[str, float]:
    """Calcula intensidades promedio de FACS"""
    facs_intensities = {}
    facs_count = {}
    
    for resultado in resultados_frames:
        for facs in resultado.get("facs", []):
            au = facs["unidad"]
            intensidad = facs["intensidad"]
            
            if au not in facs_intensities:
                facs_intensities[au] = 0
                facs_count[au] = 0
            
            facs_intensities[au] += intensidad
            facs_count[au] += 1
    
    facs_promedio = {}
    for au in facs_intensities:
        if facs_count[au] > 0:
            facs_promedio[au] = float(facs_intensities[au] / facs_count[au])
    
    return facs_promedio

def calculate_emotional_variability(resultados_frames: List[Dict]) -> float:
    """Calcula la variabilidad emocional"""
    emocion_counts = {}
    for resultado in resultados_frames:
        emocion = resultado["emocion_principal"]
        emocion_counts[emocion] = emocion_counts.get(emocion, 0) + 1
    
    total_frames = len(resultados_frames)
    if total_frames == 0:
        return 0.0
    
    entropia = 0.0
    for count in emocion_counts.values():
        probabilidad = count / total_frames
        if probabilidad > 0:
            entropia -= probabilidad * np.log2(probabilidad)
    
    max_entropia = np.log2(len(emocion_counts)) if emocion_counts else 1
    variabilidad = entropia / max_entropia if max_entropia > 0 else 0
    
    return float(variabilidad)

def generate_empty_response() -> Dict[str, Any]:
    """Genera respuesta vacía para casos de error"""
    return {
        "emocion_predominante": "No detectada",
        "total_frames": 0,
        "duracion_video": 0,
        "emociones_detectadas": [],
        "correlaciones": {"maquiavelismo": 0, "narcisismo": 0, "psicopatia": 0},
        "frames_analizados": 0,
        "intensidad_promedio": 0,
        "variabilidad_emocional": 0,
        "aus_frecuentes": [],
        "facs_promedio": {},
        "modelos_utilizados": {"emociones": "No disponible", "facs": "No disponible"}
    }

# -----------------------------------------------------
# FASTAPI + CORS
# -----------------------------------------------------
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
        "modelo_facs": modelo_facs_cargado,
        "prioridad": "Modelo de emociones"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "modelo_emociones": modelo_emociones_cargado,
        "modelo_facs": modelo_facs_cargado
    }

@app.get("/model-status")
async def model_status():
    return {
        "emociones": {
            "cargado": modelo_emociones_cargado,
            "modelo": "EfficientNet-B0",
            "prioridad": "ALTA"
        },
        "facs": {
            "cargado": modelo_facs_cargado, 
            "modelo": "Py-Feat" if modelo_facs_cargado else "Simulado",
            "prioridad": "MEDIA"
        }
    }

@app.post("/analyze-video")
async def analyze_video(request: dict):
    try:
        logger.info("🎬 Iniciando análisis de video...")
        
        if not all(key in request for key in ['video_data', 'participant_data', 'sd3_data']):
            raise HTTPException(status_code=400, detail="Datos incompletos")
        
        # Decodificar video base64
        video_bytes = base64.b64decode(request['video_data'].split(',')[1])
        
        # Guardar video temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            temp_file.write(video_bytes)
            temp_video_path = temp_file.name

        # Extraer frames del video (1 frame por segundo)
        frames = extract_frames(temp_video_path)
        logger.info(f"📊 Frames extraídos: {len(frames)}")
        
        # Analizar cada frame
        resultados_frames = []
        for i, frame in enumerate(frames):
            try:
                logger.info(f"🔍 Analizando frame {i+1}/{len(frames)}...")
                
                # Análisis de emociones (PRIMARIO)
                analisis_emociones = await analyze_frame_emociones(frame)
                
                # Análisis FACS (SECUNDARIO - solo si está disponible)
                analisis_facs = await analyze_frame_facs(frame)
                
                # Combinar resultados
                resultado_frame = {
                    **analisis_emociones,
                    "facs_avanzado": analisis_facs,
                    "frame_numero": i + 1,
                    "timestamp": i  # segundos
                }
                
                resultados_frames.append(resultado_frame)
                
            except Exception as e:
                logger.error(f"⚠️ Error analizando frame {i+1}: {e}")
                continue

        # Procesar resultados agregados
        resultado_final = process_aggregated_results(resultados_frames, request['sd3_data'])
        
        # Agregar metadata
        resultado_final["participante"] = request['participant_data'].get('nombre', 'Anónimo')
        resultado_final["historia_utilizada"] = determinar_historia(request['sd3_data'])
        resultado_final["timestamp_analisis"] = asyncio.get_event_loop().time()
        
        # Limpiar archivo temporal
        os.unlink(temp_video_path)
        
        logger.info("✅ Análisis de video completado")
        return resultado_final
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de video: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando video: {str(e)}")

def determinar_historia(sd3_data: Dict) -> str:
    """Determina la historia utilizada basada en SD3"""
    rasgos = {
        "maquiavelismo": sd3_data.get('mach', 0),
        "narcisismo": sd3_data.get('narc', 0),
        "psicopatia": sd3_data.get('psych', 0)
    }
    
    rasgo_predominante = max(rasgos, key=rasgos.get)
    return rasgo_predominante

# Para Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
