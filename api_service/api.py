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
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

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
# CONFIGURACIÓN FACS (USANDO SISTEMA SIMULADO)
# -----------------------------------------------------
modelo_facs_cargado = False
logger.info("✅ Usando sistema FACS simulado (pyfeat no compatible con Python 3.13)")

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
    """Analiza un frame usando FACS simulado"""
    return {
        "facs_disponible": False, 
        "aus": [], 
        "landmarks": [],
        "modelo_utilizado": "Simulado"
    }

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
# FUNCIONES PARA ANÁLISIS ESTADÍSTICO AVANZADO
# -----------------------------------------------------
def calculate_correlation_matrix(sd3_scores: List[Dict], facs_scores: List[Dict]) -> Dict[str, Any]:
    """Calcula matriz de correlación entre SD3 y FACS"""
    if not sd3_scores or not facs_scores:
        return {"error": "Datos insuficientes"}
    
    # Crear DataFrame
    data = []
    for sd3, facs in zip(sd3_scores, facs_scores):
        row = {
            'mach': sd3.get('mach', 0),
            'narc': sd3.get('narc', 0),
            'psych': sd3.get('psych', 0),
        }
        
        # Agregar intensidades de AUs
        if isinstance(facs, dict):
            for au, intensity in facs.items():
                row[f'AU_{au}'] = intensity
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calcular correlaciones
    correlation_matrix = df.corr()
    
    # Filtrar solo correlaciones entre SD3 y AUs
    sd3_cols = ['mach', 'narc', 'psych']
    au_cols = [col for col in df.columns if col.startswith('AU_')]
    
    correlations = {}
    for sd3_col in sd3_cols:
        correlations[sd3_col] = {}
        for au_col in au_cols:
            if sd3_col in correlation_matrix.index and au_col in correlation_matrix.columns:
                corr_value = correlation_matrix.loc[sd3_col, au_col]
                correlations[sd3_col][au_col] = float(corr_value)
    
    return {
        "matrix": correlation_matrix.to_dict(),
        "correlations": correlations,
        "significant_correlations": find_significant_correlations(correlation_matrix, sd3_cols, au_cols)
    }

def find_significant_correlations(corr_matrix: pd.DataFrame, sd3_cols: List[str], au_cols: List[str]) -> List[Dict]:
    """Encuentra correlaciones significativas (|r| > 0.3)"""
    significant = []
    
    for sd3_col in sd3_cols:
        for au_col in au_cols:
            if sd3_col in corr_matrix.index and au_col in corr_matrix.columns:
                r = corr_matrix.loc[sd3_col, au_col]
                if abs(r) > 0.3:  # Umbral de significancia
                    significant.append({
                        "sd3_trait": sd3_col,
                        "au": au_col.replace('AU_', ''),
                        "correlation": float(r),
                        "strength": "fuerte" if abs(r) > 0.5 else "moderada",
                        "direction": "positiva" if r > 0 else "negativa"
                    })
    
    # Ordenar por valor absoluto de correlación
    significant.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return significant

def perform_linear_regression(sd3_scores: List[float], facs_scores: List[float]) -> Dict[str, Any]:
    """Realiza regresión lineal simple entre SD3 y FACS"""
    if len(sd3_scores) != len(facs_scores) or len(sd3_scores) < 3:
        return {"error": "Datos insuficientes para regresión"}
    
    X = np.array(sd3_scores).reshape(-1, 1)
    y = np.array(facs_scores)
    
    # Estandarizar
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Regresión lineal
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    # Coeficientes
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    
    # Predicciones
    y_pred = model.predict(X_scaled)
    
    # Métricas
    r_squared = float(model.score(X_scaled, y_scaled))
    residuals = y_scaled - y_pred
    mse = float(np.mean(residuals**2))
    
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "mse": mse,
        "equation": f"y = {slope:.3f}x + {intercept:.3f}",
        "predictions": [float(p) for p in y_pred],
        "residuals": [float(r) for r in residuals]
    }

def generate_scatter_plot_base64(x_data: List[float], y_data: List[float], 
                                x_label: str, y_label: str, 
                                regression_line: bool = True) -> str:
    """Genera un gráfico de dispersión en base64"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(x_data, y_data, alpha=0.6, s=50, color='#7f00ff', edgecolors='white', linewidth=1)
    
    # Línea de regresión si se solicita
    if regression_line and len(x_data) > 2:
        # Calcular regresión
        X = np.array(x_data).reshape(-1, 1)
        y = np.array(y_data)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Crear línea de regresión
        x_line = np.linspace(min(x_data), max(x_data), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        
        plt.plot(x_line, y_line, color='#ff6384', linewidth=2, 
                label=f'Regresión: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}')
        
        plt.legend()
    
    # Personalizar
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f'Correlación: {x_label} vs {y_label}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convertir a base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def analyze_response_times(tiempos_data: List[Dict]) -> Dict[str, Any]:
    """Analiza tiempos de respuesta por ítem del test SD3"""
    if not tiempos_data:
        return {"error": "No hay datos de tiempos"}
    
    # Organizar tiempos por ítem
    items_analysis = {}
    
    for item_data in tiempos_data:
        item_num = item_data.get('item_number')
        if item_num:
            tiempo_ms = item_data.get('tiempo_ms', 0)
            
            if item_num not in items_analysis:
                items_analysis[item_num] = {
                    'times': [],
                    'question': f"Ítem {item_num}"
                }
            
            items_analysis[item_num]['times'].append(tiempo_ms)
    
    # Calcular estadísticas por ítem
    for item_num, data in items_analysis.items():
        times = data['times']
        if times:
            data['mean'] = float(np.mean(times))
            data['median'] = float(np.median(times))
            data['std'] = float(np.std(times))
            data['min'] = float(np.min(times))
            data['max'] = float(np.max(times))
            data['count'] = len(times)
    
    # Encontrar ítems más rápidos y más lentos
    items_list = [
        {
            'item': item_num,
            'question': data['question'],
            'mean_time': data.get('mean', 0),
            'median_time': data.get('median', 0),
            'std_time': data.get('std', 0)
        }
        for item_num, data in items_analysis.items()
    ]
    
    # Ordenar por tiempo promedio
    fastest_items = sorted(items_list, key=lambda x: x['mean_time'])[:5]
    slowest_items = sorted(items_list, key=lambda x: x['mean_time'], reverse=True)[:5]
    
    return {
        'items_analysis': items_analysis,
        'fastest_items': fastest_items,
        'slowest_items': slowest_items,
        'overall_stats': {
            'total_items': len(items_analysis),
            'avg_time_across_items': float(np.mean([item['mean_time'] for item in items_list])),
            'total_responses': sum(len(data['times']) for data in items_analysis.values())
        }
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

# -----------------------------------------------------
# NUEVOS ENDPOINTS PARA ANÁLISIS ESTADÍSTICO
# -----------------------------------------------------

@app.post("/analyze-correlations")
async def analyze_correlations(request: dict):
    """
    Analiza correlaciones entre scores SD3 y datos FACS
    """
    try:
        logger.info("📊 Analizando correlaciones SD3-FACS...")
        
        sd3_scores = request.get('sd3_scores', [])
        facs_scores = request.get('facs_scores', [])
        
        if not sd3_scores or not facs_scores:
            raise HTTPException(status_code=400, detail="Datos SD3 o FACS faltantes")
        
        # Calcular matriz de correlación
        correlation_results = calculate_correlation_matrix(sd3_scores, facs_scores)
        
        # Generar gráficos de dispersión para correlaciones significativas
        plots = {}
        significant_corrs = correlation_results.get("significant_correlations", [])
        
        for corr in significant_corrs[:3]:  # Solo primeros 3 para evitar sobrecarga
            sd3_trait = corr["sd3_trait"]
            au = corr["au"]
            
            # Extraer datos para este par
            x_data = [score.get(sd3_trait, 0) for score in sd3_scores]
            y_data = []
            
            for facs in facs_scores:
                if isinstance(facs, dict):
                    y_data.append(facs.get(au, 0))
                else:
                    y_data.append(0)
            
            # Generar gráfico
            plot_base64 = generate_scatter_plot_base64(
                x_data, y_data,
                x_label=f"SD3: {sd3_trait}",
                y_label=f"FACS: {au}",
                regression_line=True
            )
            
            plots[f"{sd3_trait}_{au}"] = plot_base64
        
        return {
            "success": True,
            "correlation_analysis": correlation_results,
            "plots": plots,
            "summary": {
                "total_correlations": len(significant_corrs),
                "strong_correlations": len([c for c in significant_corrs if c["strength"] == "fuerte"]),
                "moderate_correlations": len([c for c in significant_corrs if c["strength"] == "moderada"])
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de correlaciones: {e}")
        raise HTTPException(status_code=500, detail=f"Error analizando correlaciones: {str(e)}")

@app.post("/analyze-response-times")
async def analyze_response_times_endpoint(request: dict):
    """
    Analiza tiempos de respuesta por ítem del test SD3
    """
    try:
        logger.info("⏱️ Analizando tiempos de respuesta...")
        
        tiempos_data = request.get('tiempos_data', [])
        
        if not tiempos_data:
            raise HTTPException(status_code=400, detail="Datos de tiempos faltantes")
        
        # Analizar tiempos
        analysis_results = analyze_response_times(tiempos_data)
        
        # Generar gráfico de barras para tiempos por ítem
        if 'items_analysis' in analysis_results:
            items_analysis = analysis_results['items_analysis']
            
            # Preparar datos para gráfico
            items = []
            mean_times = []
            
            for item_num, data in sorted(items_analysis.items(), key=lambda x: x[0]):
                items.append(f"Ítem {item_num}")
                mean_times.append(data.get('mean', 0))
            
            # Crear gráfico
            plt.figure(figsize=(12, 6))
            
            colors = ['#ff6384' if t > np.mean(mean_times) else '#36a2eb' for t in mean_times]
            
            bars = plt.bar(items, mean_times, color=colors, alpha=0.7)
            plt.axhline(y=np.mean(mean_times), color='#7f00ff', linestyle='--', 
                       label=f'Promedio: {np.mean(mean_times):.0f} ms')
            
            # Añadir etiquetas
            for bar, time in zip(bars, mean_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{int(time)}', ha='center', va='bottom', fontsize=8)
            
            plt.xlabel('Ítems del Test SD3', fontsize=12)
            plt.ylabel('Tiempo Promedio (ms)', fontsize=12)
            plt.title('Tiempos de Respuesta por Ítem del Test SD3', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            # Convertir a base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            times_plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            analysis_results['times_plot'] = times_plot_base64
        
        return {
            "success": True,
            "response_time_analysis": analysis_results
        }
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de tiempos: {e}")
        raise HTTPException(status_code=500, detail=f"Error analizando tiempos: {str(e)}")

@app.post("/regression-analysis")
async def regression_analysis_endpoint(request: dict):
    """
    Realiza análisis de regresión entre variables específicas
    """
    try:
        logger.info("📈 Realizando análisis de regresión...")
        
        x_data = request.get('x_data', [])
        y_data = request.get('y_data', [])
        x_label = request.get('x_label', 'Variable X')
        y_label = request.get('y_label', 'Variable Y')
        
        if len(x_data) != len(y_data) or len(x_data) < 3:
            raise HTTPException(status_code=400, detail="Datos insuficientes para regresión")
        
        # Realizar regresión
        regression_results = perform_linear_regression(x_data, y_data)
        
        # Generar gráfico con línea de regresión
        scatter_plot = generate_scatter_plot_base64(
            x_data, y_data,
            x_label=x_label,
            y_label=y_label,
            regression_line=True
        )
        
        # Calcular estadísticas adicionales
        correlation, p_value = pearsonr(x_data, y_data)
        
        return {
            "success": True,
            "regression_results": regression_results,
            "correlation_stats": {
                "pearson_r": float(correlation),
                "p_value": float(p_value),
                "significance": "significativa" if p_value < 0.05 else "no significativa"
            },
            "scatter_plot": scatter_plot,
            "interpretation": interpret_regression_results(regression_results, correlation, p_value)
        }
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de regresión: {e}")
        raise HTTPException(status_code=500, detail=f"Error en regresión: {str(e)}")

def interpret_regression_results(regression_results: Dict, correlation: float, p_value: float) -> str:
    """Interpreta los resultados de regresión"""
    r_squared = regression_results.get('r_squared', 0)
    slope = regression_results.get('slope', 0)
    
    interpretations = []
    
    # Interpretar R²
    if r_squared > 0.7:
        interpretations.append(f"El modelo explica el {r_squared*100:.1f}% de la variabilidad (excelente ajuste)")
    elif r_squared > 0.5:
        interpretations.append(f"El modelo explica el {r_squared*100:.1f}% de la variabilidad (buen ajuste)")
    elif r_squared > 0.3:
        interpretations.append(f"El modelo explica el {r_squared*100:.1f}% de la variabilidad (ajuste moderado)")
    else:
        interpretations.append(f"El modelo explica solo el {r_squared*100:.1f}% de la variabilidad (ajuste débil)")
    
    # Interpretar pendiente
    if abs(slope) > 0.5:
        direction = "positiva" if slope > 0 else "negativa"
        strength = "fuerte" if abs(slope) > 0.8 else "moderada"
        interpretations.append(f"Relación {strength} {direction}: por cada unidad en X, Y cambia {slope:.3f} unidades")
    
    # Interpretar significancia
    if p_value < 0.001:
        interpretations.append("La correlación es altamente significativa (p < 0.001)")
    elif p_value < 0.01:
        interpretations.append("La correlación es muy significativa (p < 0.01)")
    elif p_value < 0.05:
        interpretations.append("La correlación es significativa (p < 0.05)")
    else:
        interpretations.append("La correlación no es estadísticamente significativa")
    
    return " | ".join(interpretations)

@app.get("/advanced-stats-summary")
async def advanced_stats_summary():
    """
    Devuelve un resumen de las capacidades estadísticas avanzadas
    """
    return {
        "available_analyses": [
            {
                "name": "Correlaciones SD3-FACS",
                "endpoint": "/analyze-correlations",
                "description": "Analiza correlaciones entre rasgos de personalidad y unidades de acción facial",
                "method": "POST",
                "required_data": ["sd3_scores", "facs_scores"]
            },
            {
                "name": "Análisis de Tiempos de Respuesta",
                "endpoint": "/analyze-response-times",
                "description": "Analiza tiempos de respuesta por ítem del test SD3",
                "method": "POST",
                "required_data": ["tiempos_data"]
            },
            {
                "name": "Regresión Lineal",
                "endpoint": "/regression-analysis",
                "description": "Realiza análisis de regresión lineal entre dos variables",
                "method": "POST",
                "required_data": ["x_data", "y_data", "x_label", "y_label"]
            }
        ],
        "statistical_methods": [
            "Correlación de Pearson",
            "Regresión Lineal Simple",
            "Análisis de significancia estadística",
            "Generación de gráficos de dispersión",
            "Análisis de tiempos de respuesta"
        ]
    }

# Para Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
