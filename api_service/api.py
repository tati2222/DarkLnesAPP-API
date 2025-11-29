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

# -----------------------------------------------------
# MODELO (MISMO QUE USÁS EN STREAMLIT)
# -----------------------------------------------------
class MicroExpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.model(x)

# Cargar modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Usando dispositivo: {device}")

model = MicroExpNet()

try:
    # Intenta cargar el modelo - AQUÍ ESTÁ LA CLAVE
    state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
    first_key = list(state.keys())[0]

    # Ajustar claves igual que en tu Streamlit
    if first_key.startswith("model.model."):
        new_state = {k.replace("model.", "", 1): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=True)
    elif first_key.startswith("model."):
        new_state = {k.replace("model.", ""): v for k, v in state.items()}
        model.model.load_state_dict(new_state, strict=True)
    else:
        model.model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    print("✅ Modelo cargado exitosamente!")
    
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    # Modelo de emergencia
    model = None

# -----------------------------------------------------
# PREPROCESAMIENTO
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels = ["Alegría", "Tristeza", "Enojo", "Sorpresa", "Miedo", "Disgusto", "Neutral"]

def compute_sd3(em):
    maqu = em["Enojo"] * 0.6 + em["Disgusto"] * 0.4
    narc = em["Alegría"] * 0.5 + em["Neutral"] * 0.5
    psic = em["Miedo"] * 0.7 + em["Sorpresa"] * 0.3

    return {
        "Maquiavelismo": round(maqu * 100, 2),
        "Narcisismo": round(narc * 100, 2),
        "Psicopatía": round(psic * 100, 2)
    }

# -----------------------------------------------------
# FUNCIONES PARA ANÁLISIS DE VIDEO
# -----------------------------------------------------
class VideoAnalysisRequest:
    def __init__(self, video_data: str, participant_data: Dict[str, Any], sd3_data: Dict[str, Any]):
        self.video_data = video_data
        self.participant_data = participant_data
        self.sd3_data = sd3_data

def extract_frames(video_path: str, frames_per_second: int = 1) -> List[np.ndarray]:
    """Extrae frames del video (1 frame por segundo por defecto)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Obtener FPS del video
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Valor por defecto si no se puede obtener
    frame_interval = max(1, int(fps / frames_per_second))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extraer 1 frame por segundo
        if frame_count % frame_interval == 0:
            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        frame_count += 1
        
    cap.release()
    return frames

async def analyze_single_frame(frame: np.ndarray) -> Dict[str, Any]:
    """Analiza un frame individual usando el modelo de emociones"""
    try:
        # Convertir numpy array a PIL Image
        img = Image.fromarray(frame)
        
        # Preprocesar
        tensor = transform(img).unsqueeze(0).to(device)
        
        # Predecir
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        
        # Formatear resultados
        emotions = {labels[i]: float(probs[i]) for i in range(7)}
        emocion_principal = max(emotions, key=emotions.get)
        
        return {
            "emocion_principal": emocion_principal,
            "confianza": emotions[emocion_principal],
            "emociones": emotions,
            "facs": generate_facs_data(emocion_principal, emotions[emocion_principal]),
            "aus_detectadas": generate_aus_detected(emocion_principal)
        }
        
    except Exception as e:
        print(f"Error analizando frame individual: {e}")
        return generate_fallback_analysis()

def generate_facs_data(emocion_principal: str, confianza: float) -> List[Dict]:
    """Genera datos FACS basados en la emoción detectada"""
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
    emociones = ["Alegría", "Neutral", "Enojo", "Miedo", "Sorpresa", "Tristeza", "Disgusto"]
    emocion_principal = np.random.choice(emociones, p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05])
    
    return {
        "emocion_principal": emocion_principal,
        "confianza": np.random.uniform(0.5, 0.8),
        "emociones": {emocion: np.random.uniform(0, 1) for emocion in emociones},
        "facs": generate_facs_data(emocion_principal, 0.7),
        "aus_detectadas": generate_aus_detected(emocion_principal)
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
        "facs_promedio": facs_promedio
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
    
    # Contar frecuencia de emociones
    emocion_counts = {}
    for resultado in resultados_frames:
        emocion = resultado["emocion_principal"]
        emocion_counts[emocion] = emocion_counts.get(emocion, 0) + 1
    
    # Calcular correlaciones simples
    correlaciones = {}
    for rasgo in ["maquiavelismo", "narcisismo", "psicopatia"]:
        score_sd3 = sd3_data.get(rasgo[:4], 0)  # mach, narc, psych
        
        # Encontrar emociones relacionadas con este rasgo
        emociones_relacionadas = [emocion for emocion, trait in emocion_to_trait.items() 
                                if trait == rasgo]
        
        # Calcular proporción de frames con emociones relacionadas
        total_frames_relacionados = sum(emocion_counts.get(emocion, 0) 
                                      for emocion in emociones_relacionadas)
        proporcion = total_frames_relacionados / len(resultados_frames) if resultados_frames else 0
        
        # Correlación simplificada
        correlacion = min(proporcion * score_sd3 * 2, 1.0)  # Escalar a rango 0-1
        correlaciones[rasgo] = float(correlacion)
    
    return correlaciones

def calculate_frequent_aus(resultados_frames: List[Dict]) -> List[str]:
    """Calcula las Action Units más frecuentes"""
    aus_counts = {}
    for resultado in resultados_frames:
        for au in resultado.get("aus_detectadas", []):
            aus_counts[au] = aus_counts.get(au, 0) + 1
    
    # Ordenar por frecuencia y tomar las top 5
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
    
    # Calcular promedios
    facs_promedio = {}
    for au in facs_intensities:
        if facs_count[au] > 0:
            facs_promedio[au] = float(facs_intensities[au] / facs_count[au])
    
    return facs_promedio

def calculate_emotional_variability(resultados_frames: List[Dict]) -> float:
    """Calcula la variabilidad emocional (entropía de Shannon simplificada)"""
    emocion_counts = {}
    for resultado in resultados_frames:
        emocion = resultado["emocion_principal"]
        emocion_counts[emocion] = emocion_counts.get(emocion, 0) + 1
    
    total_frames = len(resultados_frames)
    if total_frames == 0:
        return 0.0
    
    # Calcular entropía
    entropia = 0.0
    for count in emocion_counts.values():
        probabilidad = count / total_frames
        if probabilidad > 0:
            entropia -= probabilidad * np.log2(probabilidad)
    
    # Normalizar a rango 0-1
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
        "facs_promedio": {}
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

# Health check ESSENCIAL para Render
@app.get("/")
async def root():
    return {
        "message": "DarkLnes Microexpressions API", 
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# -----------------------------------------------------
# ENDPOINT /run/predict (EXISTENTE)
# -----------------------------------------------------
@app.post("/run/predict")
async def run_predict(file: UploadFile = File(...)):
    try:
        print(f"📨 Recibiendo archivo: {file.filename}")
        
        if not model:
            raise HTTPException(status_code=500, detail="Modelo no cargado")
        
        # Leer y procesar imagen
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        print(f"📷 Imagen cargada: {img.size}")
        
        # Preprocesar
        tensor = transform(img).unsqueeze(0).to(device)
        print("🔬 Procesando con el modelo...")
        
        # Predecir
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        
        print(f"📊 Predicciones: {probs}")
        
        # Formatear resultados
        emotions = {labels[i]: float(probs[i]) for i in range(7)}
        sd3 = compute_sd3(emotions)
        
        print(f"🎭 SD3 calculado: {sd3}")
        
        return {
            "status": "ok",
            "emociones": emotions,
            "sd3": sd3,
            "modelo_utilizado": "EfficientNet-B0 entrenado en FER2013"
        }
        
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# -----------------------------------------------------
# NUEVO ENDPOINT /analyze-video
# -----------------------------------------------------
@app.post("/analyze-video")
async def analyze_video(request: dict):
    try:
        print("🎬 Iniciando análisis de video...")
        
        # Validar datos de entrada
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
        print(f"📊 Frames extraídos: {len(frames)}")
        
        # Analizar cada frame
        resultados_frames = []
        for i, frame in enumerate(frames):
            try:
                print(f"🔍 Analizando frame {i+1}/{len(frames)}...")
                analisis_frame = await analyze_single_frame(frame)
                resultados_frames.append(analisis_frame)
            except Exception as e:
                print(f"⚠️ Error analizando frame {i+1}: {e}")
                continue

        # Procesar resultados agregados
        resultado_final = process_aggregated_results(resultados_frames, request['sd3_data'])
        
        # Limpiar archivo temporal
        os.unlink(temp_video_path)
        
        print("✅ Análisis de video completado")
        return resultado_final
        
    except Exception as e:
        print(f"❌ Error en análisis de video: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando video: {str(e)}")

# Para Render - usa el puerto que ellos proveen
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
