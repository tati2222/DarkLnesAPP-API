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
import pandas as pd
from scipy.stats import linregress, pearsonr
import matplotlib
matplotlib.use('Agg')  # Para usar matplotlib en entornos sin display
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# =====================================================
# CONFIGURACIÓN SUPABASE
# =====================================================
try:
    from supabase import create_client, Client
    
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
    
    if SUPABASE_URL and SUPABASE_KEY:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase_available = True
    else:
        supabase_available = False
        print("⚠️ Supabase no configurado. Variables de entorno faltantes.")
        
except ImportError:
    print("⚠️ Biblioteca supabase no instalada. Ejecuta: pip install supabase")
    supabase_available = False
except Exception as e:
    print(f"❌ Error configurando Supabase: {e}")
    supabase_available = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# EXPLICACIÓN ESCALA SD3
# =====================================================
"""
ESCALA DEL TEST SD3 (TRIADA OSCURA):
1 = Totalmente en desacuerdo
2 = En desacuerdo  
3 = Neutral / Ni de acuerdo ni en desacuerdo
4 = De acuerdo
5 = Totalmente de acuerdo

Los resultados se interpretan en una escala continua de 1 a 5,
donde valores más altos indican mayor presencia del rasgo.
"""

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
# FUNCIONES PARA VALIDACIÓN Y NORMALIZACIÓN SD3
# =====================================================
def validate_sd3_scores(sd3_data: Dict) -> Dict:
    """
    Valida y normaliza los puntajes SD3.
    Escala original: 1-5 donde 1 es "Totalmente en desacuerdo" y 5 "Totalmente de acuerdo"
    Normalizamos a 0-1 para cálculos internos.
    """
    validated = {}
    
    for trait, score in sd3_data.items():
        if trait in ['mach', 'narc', 'psych']:
            try:
                # Asegurar que el valor esté en el rango 1-5
                score_float = float(score)
                if score_float < 1:
                    logger.warning(f"⚠️ Puntaje {trait} muy bajo ({score_float}), ajustando a 1")
                    score_float = 1.0
                elif score_float > 5:
                    logger.warning(f"⚠️ Puntaje {trait} muy alto ({score_float}), ajustando a 5")
                    score_float = 5.0
                
                # Normalizar a rango 0-1 (donde 0 = mínimo rasgo, 1 = máximo rasgo)
                normalized = (score_float - 1) / 4.0  # (5-1=4)
                validated[trait] = normalized
                validated[f"{trait}_raw"] = score_float  # Guardar valor original
                
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Error procesando puntaje {trait}: {score} - Error: {e}")
                validated[trait] = 0.0
                validated[f"{trait}_raw"] = 0.0
        else:
            validated[trait] = score
    
    return validated

def get_sd3_interpretation(sd3_scores: Dict) -> Dict:
    """
    Proporciona interpretación de los puntajes SD3 normalizados
    """
    interpretations = {}
    
    for trait, normalized_score in sd3_scores.items():
        if trait in ['mach', 'narc', 'psych']:
            raw_score = sd3_scores.get(f"{trait}_raw", (normalized_score * 4) + 1)
            
            # Interpretación basada en el puntaje original (1-5)
            if raw_score <= 1.5:
                level = "Muy bajo"
                description = f"Poco {trait} presente"
            elif raw_score <= 2.5:
                level = "Bajo"
                description = f"{trait} presente en grado bajo"
            elif raw_score <= 3.5:
                level = "Moderado"
                description = f"{trait} presente en grado moderado"
            elif raw_score <= 4.5:
                level = "Alto"
                description = f"{trait} presente en grado alto"
            else:
                level = "Muy alto"
                description = f"{trait} muy marcado"
            
            interpretations[trait] = {
                "puntaje_original": float(raw_score),
                "puntaje_normalizado": float(normalized_score),
                "nivel": level,
                "descripcion": description,
                "interpretacion": get_trait_interpretation(trait, normalized_score)
            }
    
    return interpretations

def get_trait_interpretation(trait: str, score: float) -> str:
    """Proporciona interpretación detallada de cada rasgo"""
    interpretations = {
        "mach": {
            "low": "Baja tendencia a manipular o usar a otros para beneficio personal.",
            "medium": "Capacidad moderada para influenciar situaciones sociales cuando es necesario.",
            "high": "Fuerte tendencia a usar estrategias calculadas para alcanzar objetivos personales."
        },
        "narc": {
            "low": "Baja necesidad de atención y validación externa.",
            "medium": "Autoestima equilibrada, con confianza pero sin necesidad excesiva de admiración.",
            "high": "Fuerte necesidad de admiración y reconocimiento por parte de otros."
        },
        "psych": {
            "low": "Alta empatía y consideración por los sentimientos de los demás.",
            "medium": "Empatía moderada, capaz de tomar decisiones difíciles cuando es necesario.",
            "high": "Baja empatía y tendencia a actuar impulsivamente sin considerar consecuencias."
        }
    }
    
    if score < 0.33:
        return interpretations[trait]["low"]
    elif score < 0.66:
        return interpretations[trait]["medium"]
    else:
        return interpretations[trait]["high"]

# =====================================================
# FUNCIONES SUPABASE - ADAPTADAS A darklens_records
# =====================================================
async def upload_image_to_supabase(image_base64: str, analysis_id: str) -> str:
    """Sube imagen a Supabase Storage y retorna URL pública"""
    try:
        if not supabase_available:
            return None
        
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        filename = f"{analysis_id}.jpg"
        
        try:
            supabase_client.storage.from_("analisis-images").upload(
                filename,
                image_bytes,
                {"content-type": "image/jpeg"}
            )
            
            image_url = supabase_client.storage.from_("analisis-images").get_public_url(filename)
            logger.info(f"✅ Imagen subida a Supabase Storage: {image_url}")
            return image_url
            
        except Exception as storage_error:
            logger.warning(f"Bucket no encontrado: {storage_error}")
            return f"storage/analisis-images/{filename}"
                
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
    """
    try:
        analysis_id = str(uuid.uuid4())
        
        image_url = None
        if image_base64 and supabase_available:
            image_url = await upload_image_to_supabase(image_base64, analysis_id)
        
        # Usar valores RAW (1-5) para guardar en Supabase
        record_data = {
            "id": analysis_id,
            "nombre": participant_data.get('nombre', 'Anónimo'),
            "edad": participant_data.get('edad', None),
            "genero": participant_data.get('genero', None),
            "pais": participant_data.get('pais', None),
            "mach": float(sd3_data.get('mach_raw', sd3_data.get('mach', 0.0) * 4 + 1)),
            "narc": float(sd3_data.get('narc_raw', sd3_data.get('narc', 0.0) * 4 + 1)),
            "psych": float(sd3_data.get('psych_raw', sd3_data.get('psych', 0.0) * 4 + 1)),
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
        
        if supabase_available:
            try:
                response = supabase_client.table("darklens_records").insert(record_data).execute()
                
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
                return await save_to_local_fallback(record_data, image_base64, analysis_id)
        else:
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
        os.makedirs("local_storage", exist_ok=True)
        
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

async def get_research_data_from_supabase() -> List[Dict]:
    """Obtiene todos los datos de investigación de Supabase"""
    try:
        if not supabase_available:
            return []
        
        # Obtener todos los registros
        response = supabase_client.table("darklens_records").select("*").execute()
        
        if hasattr(response, 'data'):
            return response.data
        elif isinstance(response, dict) and 'data' in response:
            return response['data']
        else:
            return []
            
    except Exception as e:
        logger.error(f"❌ Error obteniendo datos de investigación: {e}")
        return []

# =====================================================
# FUNCIONES DE ANÁLISIS ESTADÍSTICO PARA INVESTIGADOR
# =====================================================
def analyze_sd3_facs_correlation(records: List[Dict]) -> Dict:
    """
    Realiza análisis de correlación entre puntajes SD3 y datos FACS
    """
    if not records or len(records) < 5:
        return {"error": "Se necesitan al menos 5 registros para análisis estadístico"}
    
    try:
        # Preparar DataFrame
        data = []
        for record in records:
            try:
                # Datos SD3
                mach = float(record.get('mach', 0))
                narc = float(record.get('narc', 0))
                psych = float(record.get('psych', 0))
                
                # Datos FACS (del JSON almacenado)
                facs_json = record.get('facs_promedio', '{}')
                if isinstance(facs_json, str):
                    facs_data = json.loads(facs_json)
                else:
                    facs_data = facs_json
                
                # Emoción detectada
                emocion = record.get('emocion_princ', 'Neutral')
                
                # Crear registro
                record_data = {
                    'id': record.get('id', ''),
                    'mach': mach,
                    'narc': narc,
                    'psych': psych,
                    'emocion': emocion,
                    'total_sd3': mach + narc + psych,
                    'edad': record.get('edad', 0),
                    'genero': record.get('genero', ''),
                    'pais': record.get('pais', '')
                }
                
                # Agregar datos FACS
                for au, intensity in facs_data.items():
                    record_data[f'facs_{au}'] = float(intensity)
                
                data.append(record_data)
                
            except Exception as e:
                logger.warning(f"⚠️ Error procesando registro: {e}")
                continue
        
        if len(data) < 5:
            return {"error": "Datos insuficientes después de procesamiento"}
        
        df = pd.DataFrame(data)
        
        # Análisis de correlación
        correlations = {}
        
        # 1. Correlación SD3 vs FACS
        facs_columns = [col for col in df.columns if col.startswith('facs_')]
        sd3_traits = ['mach', 'narc', 'psych', 'total_sd3']
        
        for trait in sd3_traits:
            if trait in df.columns:
                trait_correlations = {}
                for facs_col in facs_columns:
                    if facs_col in df.columns:
                        # Filtrar valores no nulos
                        valid_data = df[[trait, facs_col]].dropna()
                        if len(valid_data) >= 3:
                            corr, p_value = pearsonr(valid_data[trait], valid_data[facs_col])
                            trait_correlations[facs_col] = {
                                'correlacion': float(corr),
                                'p_valor': float(p_value),
                                'significativo': p_value < 0.05,
                                'n': len(valid_data)
                            }
                
                # Ordenar por correlación más fuerte
                sorted_corrs = sorted(
                    trait_correlations.items(),
                    key=lambda x: abs(x[1]['correlacion']),
                    reverse=True
                )[:10]  # Top 10 correlaciones
                
                correlations[trait] = {
                    'todas': trait_correlations,
                    'top_positivas': [item for item in sorted_corrs if item[1]['correlacion'] > 0],
                    'top_negativas': [item for item in sorted_corrs if item[1]['correlacion'] < 0],
                    'significativas': [item for item in trait_correlations.items() if item[1]['significativo']]
                }
        
        # 2. Análisis de regresión lineal para correlaciones significativas
        regresiones = {}
        for trait in sd3_traits:
            if trait in df.columns and f'correlaciones_{trait}' in correlations:
                regresiones[trait] = {}
                for facs_col, corr_data in correlations[trait]['significativas']:
                    if corr_data['significativo']:
                        valid_data = df[[trait, facs_col]].dropna()
                        if len(valid_data) >= 3:
                            slope, intercept, r_value, p_value, std_err = linregress(
                                valid_data[trait], valid_data[facs_col]
                            )
                            regresiones[trait][facs_col] = {
                                'pendiente': float(slope),
                                'intercepto': float(intercept),
                                'r_cuadrado': float(r_value**2),
                                'p_valor': float(p_value),
                                'error_estandar': float(std_err),
                                'ecuacion': f"y = {slope:.3f}x + {intercept:.3f}"
                            }
        
        # 3. Estadísticas descriptivas
        estadisticas = {
            'total_registros': len(df),
            'estadisticas_sd3': {
                'mach': {
                    'media': float(df['mach'].mean()),
                    'mediana': float(df['mach'].median()),
                    'desviacion': float(df['mach'].std()),
                    'min': float(df['mach'].min()),
                    'max': float(df['mach'].max())
                },
                'narc': {
                    'media': float(df['narc'].mean()),
                    'mediana': float(df['narc'].median()),
                    'desviacion': float(df['narc'].std()),
                    'min': float(df['narc'].min()),
                    'max': float(df['narc'].max())
                },
                'psych': {
                    'media': float(df['psych'].mean()),
                    'mediana': float(df['psych'].median()),
                    'desviacion': float(df['psych'].std()),
                    'min': float(df['psych'].min()),
                    'max': float(df['psych'].max())
                }
            },
            'distribucion_emociones': df['emocion'].value_counts().to_dict(),
            'distribucion_genero': df['genero'].value_counts().to_dict() if 'genero' in df.columns else {}
        }
        
        # 4. Interpretación de resultados
        interpretacion = interpret_correlation_results(correlations, regresiones, estadisticas)
        
        return {
            'estadisticas_descriptivas': estadisticas,
            'correlaciones': correlations,
            'regresiones': regresiones,
            'interpretacion': interpretacion,
            'dataframe_info': {
                'filas': len(df),
                'columnas': list(df.columns),
                'muestra_preview': df.head(5).to_dict('records')
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de correlación: {e}")
        return {"error": f"Error en análisis: {str(e)}"}

def interpret_correlation_results(correlations: Dict, regresiones: Dict, estadisticas: Dict) -> Dict:
    """Interpreta los resultados de correlación"""
    interpretacion = {
        'resumen': {},
        'hallazgos_significativos': [],
        'recomendaciones': []
    }
    
    # Interpretar cada rasgo SD3
    for trait in ['mach', 'narc', 'psych', 'total_sd3']:
        if trait in correlations:
            trait_name = {
                'mach': 'Maquiavelismo',
                'narc': 'Narcisismo',
                'psych': 'Psicopatía',
                'total_sd3': 'Triada Oscura Total'
            }.get(trait, trait)
            
            # Contar correlaciones significativas
            sig_count = len(correlations[trait]['significativas'])
            total_count = len(correlations[trait]['todas'])
            
            # Encontrar correlaciones más fuertes
            if correlations[trait]['top_positivas']:
                top_pos = correlations[trait]['top_positivas'][0]
                interpretacion['hallazgos_significativos'].append(
                    f"🔵 {trait_name} correlaciona positivamente con {top_pos[0]} (r={top_pos[1]['correlacion']:.2f})"
                )
            
            if correlations[trait]['top_negativas']:
                top_neg = correlations[trait]['top_negativas'][0]
                interpretacion['hallazgos_significativos'].append(
                    f"🔴 {trait_name} correlaciona negativamente con {top_neg[0]} (r={top_neg[1]['correlacion']:.2f})"
                )
            
            interpretacion['resumen'][trait_name] = {
                'correlaciones_significativas': sig_count,
                'correlaciones_totales': total_count,
                'porcentaje_significativas': f"{(sig_count/total_count*100):.1f}%" if total_count > 0 else "0%"
            }
    
    # Recomendaciones basadas en los hallazgos
    if interpretacion['hallazgos_significativos']:
        interpretacion['recomendaciones'].append(
            "✅ Considerar estas correlaciones para hipótesis de investigación futura"
        )
    else:
        interpretacion['recomendaciones'].append(
            "⚠️ No se encontraron correlaciones significativas con el tamaño de muestra actual"
        )
    
    interpretacion['recomendaciones'].append(
        f"📊 Tamaño de muestra: {estadisticas['total_registros']} participantes"
    )
    
    return interpretacion

def generate_correlation_plot(x_data: List[float], y_data: List[float], 
                             x_label: str, y_label: str, title: str) -> str:
    """Genera un gráfico de correlación y lo devuelve en base64"""
    try:
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(x_data, y_data, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # Línea de regresión
        if len(x_data) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
            line_x = np.array([min(x_data), max(x_data)])
            line_y = slope * line_x + intercept
            
            plt.plot(line_x, line_y, 'r-', label=f'y = {slope:.2f}x + {intercept:.2f}')
            plt.text(0.05, 0.95, f'r = {r_value:.2f}, p = {p_value:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Guardar en buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convertir a base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"❌ Error generando gráfico: {e}")
        return None

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

def calculate_correlations_single(resultado_frame: Dict, sd3_data_normalized: Dict) -> Dict[str, float]:
    """
    Calcula correlaciones entre una emoción y rasgos SD3
    Usa valores normalizados (0-1) donde 0 = mínimo rasgo, 1 = máximo rasgo
    """
    # Mapeo emociones -> rasgos (basado en investigación psicológica)
    emocion_to_trait = {
        "Alegría": "narcisismo",      # Alegría excesiva puede relacionarse con narcisismo
        "Enojo": "maquiavelismo",     # Enojo/irritabilidad se asocia con manipulación
        "Miedo": "psicopatia",        # Ausencia de miedo es característica de psicopatía
        "Neutral": "narcisismo",      # Neutralidad puede ser control emocional narcisista
        "Tristeza": "psicopatia",     # Ausencia de tristeza en psicopatía
        "Sorpresa": "narcisismo",     # Necesidad de atención
        "Disgusto": "maquiavelismo"   # Desprecio hacia otros
    }
    
    emocion = resultado_frame["emocion_principal"]
    confianza = resultado_frame["confianza"]
    
    correlaciones = {}
    for rasgo in ["maquiavelismo", "narcisismo", "psicopatia"]:
        # Usar valor normalizado (0-1)
        score_sd3 = sd3_data_normalized.get(rasgo, 0)
        
        rasgo_relacionado = emocion_to_trait.get(emocion, "narcisismo")
        
        if rasgo == rasgo_relacionado:
            # Si la emoción está relacionada con este rasgo
            correlacion = min(confianza * score_sd3 * 1.5, 1.0)
        else:
            # Si no está relacionada, correlación más baja
            correlacion = min((1 - confianza) * score_sd3 * 0.3, 0.3)
        
        correlaciones[rasgo] = float(correlacion)
    
    return correlaciones

def determinar_historia(sd3_data: Dict) -> str:
    """Determina la historia utilizada basada en SD3 usando valores originales (1-5)"""
    # Usar valores RAW si existen, de lo contrario calcular
    mach_raw = sd3_data.get('mach_raw', sd3_data.get('mach', 0) * 4 + 1)
    narc_raw = sd3_data.get('narc_raw', sd3_data.get('narc', 0) * 4 + 1)
    psych_raw = sd3_data.get('psych_raw', sd3_data.get('psych', 0) * 4 + 1)
    
    rasgos = {
        "maquiavelismo": mach_raw,
        "narcisismo": narc_raw,
        "psicopatia": psych_raw
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
        "explicacion_sd3": {
            "escala": "1-5 donde 1 = Totalmente en desacuerdo, 5 = Totalmente de acuerdo",
            "normalizacion": "Los valores se normalizan a 0-1 para análisis",
            "interpretacion": "Valores más altos indican mayor presencia del rasgo"
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

@app.get("/sd3-info")
async def sd3_info():
    """Endpoint para obtener información sobre la escala SD3"""
    return {
        "test_name": "Short Dark Triad (SD3)",
        "scale": {
            "range": "1-5",
            "1": "Totalmente en desacuerdo",
            "2": "En desacuerdo",
            "3": "Neutral / Ni de acuerdo ni en desacuerdo", 
            "4": "De acuerdo",
            "5": "Totalmente de acuerdo"
        },
        "traits": {
            "mach": "Maquiavelismo: Tendencia a manipular y usar a otros",
            "narc": "Narcisismo: Necesidad de admiración y sentido de grandiosidad",
            "psych": "Psicopatía: Impulsividad y falta de empatía"
        },
        "interpretation": "Puntajes más altos indican mayor presencia del rasgo"
    }

@app.post("/analyze-image")
async def analyze_image(request: dict):
    try:
        logger.info("🖼️ Iniciando análisis de imagen...")
        
        required_keys = ['image_data', 'participant_data', 'sd3_data']
        if not all(key in request for key in required_keys):
            raise HTTPException(status_code=400, detail=f"Datos incompletos. Requeridos: {required_keys}")
        
        # Validar y normalizar puntajes SD3
        logger.info("📊 Validando puntajes SD3...")
        sd3_validated = validate_sd3_scores(request['sd3_data'])
        
        # Obtener interpretación de los puntajes
        sd3_interpretation = get_sd3_interpretation(sd3_validated)
        
        # Decodificar imagen
        logger.info("📥 Decodificando imagen...")
        frame = decode_image_base64(request['image_data'])
        
        # Analizar imagen
        logger.info("🔍 Analizando imagen...")
        analisis_emociones = await analyze_frame_emociones(frame)
        
        # Procesar resultados
        emocion_principal = analisis_emociones["emocion_principal"]
        confianza = analisis_emociones["confianza"]
        
        # Calcular correlaciones con valores normalizados
        correlaciones = calculate_correlations_single(analisis_emociones, sd3_validated)
        
        # Preparar resultados
        analysis_results = {
            "emocion_principal": emocion_principal,
            "confianza": confianza,
            "total_frames": 1,
            "duracion_analisis": 0.0,
            "emociones_detectadas": [emocion_principal],
            "correlaciones": correlaciones,
            "correlaciones_explicadas": {
                "maquiavelismo": f"Correlación: {correlaciones.get('maquiavelismo', 0):.2f}. {emocion_principal} puede relacionarse con tendencias manipulativas.",
                "narcisismo": f"Correlación: {correlaciones.get('narcisismo', 0):.2f}. {emocion_principal} puede asociarse con necesidades de atención.",
                "psicopatia": f"Correlación: {correlaciones.get('psicopatia', 0):.2f}. {emocion_principal} puede vincularse con respuestas emocionales atípicas."
            },
            "frames_analizados": 1,
            "intensidad_promedio": float(confianza),
            "variabilidad_emocional": 0.0,
            "aus_frecuentes": analisis_emociones.get("aus_detectadas", []),
            "facs_promedio": {f["unidad"]: f["intensidad"] for f in analisis_emociones.get("facs", [])},
            "historia_utilizada": determinar_historia(sd3_validated),
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
            sd3_data=sd3_validated,  # Usar datos validados
            analysis_results=analysis_results,
            image_base64=request['image_data']
        )
        
        # Construir respuesta final
        response_data = {
            "analisis_facial": analysis_results,
            "participante": {
                **request['participant_data'],
                "test_completado": True
            },
            "test_sd3": {
                "puntajes_originales": {
                    "maquiavelismo": sd3_validated.get('mach_raw', 0),
                    "narcisismo": sd3_validated.get('narc_raw', 0),
                    "psicopatia": sd3_validated.get('psych_raw', 0)
                },
                "puntajes_normalizados": {
                    "maquiavelismo": sd3_validated.get('mach', 0),
                    "narcisismo": sd3_validated.get('narc', 0),
                    "psicopatia": sd3_validated.get('psych', 0)
                },
                "interpretacion": sd3_interpretation,
                "escala_explicada": {
                    "rango": "1-5",
                    "1": "Totalmente en desacuerdo",
                    "2": "En desacuerdo",
                    "3": "Neutral",
                    "4": "De acuerdo", 
                    "5": "Totalmente de acuerdo"
                }
            },
            "correlacion_resultados": {
                "resumen": f"Emoción detectada: {emocion_principal} (confianza: {confianza:.2f}).",
                "rasgo_predominante": determinar_historia(sd3_validated),
                "implicaciones": "Nota: Las correlaciones son indicativas y no diagnósticas."
            },
            "metadata": {
                "timestamp_analisis": datetime.utcnow().isoformat(),
                "tipo_analisis": "imagen",
                "storage_info": save_result,
                "modelo_version": "EfficientNet-B0 FER2013"
            }
        }
        
        logger.info("✅ Análisis completado exitosamente")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en análisis de imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

# =====================================================
# ENDPOINTS PARA PANEL DE INVESTIGADOR
# =====================================================

@app.get("/research/correlation-analysis")
async def research_correlation_analysis():
    """Endpoint principal para análisis de correlación"""
    try:
        logger.info("📊 Iniciando análisis de correlación para investigación...")
        
        # Obtener datos de Supabase
        records = await get_research_data_from_supabase()
        
        if not records:
            return {
                "status": "error",
                "message": "No hay datos disponibles para análisis",
                "recomendacion": "Espere a que más participantes completen el test"
            }
        
        # Realizar análisis
        analysis = analyze_sd3_facs_correlation(records)
        
        if "error" in analysis:
            return {
                "status": "error",
                "message": analysis["error"],
                "total_registros": len(records)
            }
        
        logger.info(f"✅ Análisis completado con {len(records)} registros")
        
        return {
            "status": "success",
            "total_registros": len(records),
            "analisis": analysis,
            "fecha_analisis": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de investigación: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")

@app.get("/research/correlation-plot/{trait}/{au}")
async def research_correlation_plot(trait: str, au: str):
    """Genera gráfico de correlación específico"""
    try:
        # Obtener datos
        records = await get_research_data_from_supabase()
        
        if not records or len(records) < 3:
            raise HTTPException(status_code=400, detail="Datos insuficientes para gráfico")
        
        # Extraer datos para el gráfico
        x_data = []
        y_data = []
        
        for record in records:
            try:
                # Valor SD3
                trait_value = float(record.get(trait, 0))
                
                # Valor FACS
                facs_json = record.get('facs_promedio', '{}')
                if isinstance(facs_json, str):
                    facs_data = json.loads(facs_json)
                else:
                    facs_data = facs_json
                
                facs_value = facs_data.get(au, 0)
                if isinstance(facs_value, (int, float)):
                    y_data.append(float(facs_value))
                    x_data.append(trait_value)
                    
            except Exception as e:
                continue
        
        if len(x_data) < 3:
            raise HTTPException(status_code=400, detail="Datos insuficientes después de filtrado")
        
        # Generar gráfico
        trait_names = {
            'mach': 'Maquiavelismo',
            'narc': 'Narcisismo',
            'psych': 'Psicopatía'
        }
        
        plot_base64 = generate_correlation_plot(
            x_data=x_data,
            y_data=y_data,
            x_label=f"{trait_names.get(trait, trait)} (1-5)",
            y_label=f"Intensidad {au}",
            title=f"Correlación: {trait_names.get(trait, trait)} vs {au}"
        )
        
        if not plot_base64:
            raise HTTPException(status_code=500, detail="Error generando gráfico")
        
        # Calcular estadísticas
        if len(x_data) >= 2:
            corr, p_value = pearsonr(x_data, y_data)
            slope, intercept, r_value, p_lin, std_err = linregress(x_data, y_data)
            
            estadisticas = {
                'correlacion_pearson': float(corr),
                'p_valor': float(p_value),
                'r_cuadrado': float(r_value**2),
                'ecuacion_regresion': f"y = {slope:.3f}x + {intercept:.3f}",
                'n': len(x_data)
            }
        else:
            estadisticas = {}
        
        return {
            "status": "success",
            "plot": plot_base64,
            "estadisticas": estadisticas,
            "datos": {
                "trait": trait,
                "au": au,
                "muestra": len(x_data)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generando gráfico de correlación: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando gráfico: {str(e)}")

@app.get("/research/descriptive-stats")
async def research_descriptive_stats():
    """Estadísticas descriptivas de todos los datos"""
    try:
        records = await get_research_data_from_supabase()
        
        if not records:
            return {"status": "error", "message": "No hay datos disponibles"}
        
        # Convertir a DataFrame para análisis
        data = []
        for record in records:
            data.append({
                'id': record.get('id'),
                'mach': float(record.get('mach', 0)),
                'narc': float(record.get('narc', 0)),
                'psych': float(record.get('psych', 0)),
                'edad': float(record.get('edad', 0)) if record.get('edad') else None,
                'emocion': record.get('emocion_princ', ''),
                'pais': record.get('pais', ''),
                'genero': record.get('genero', '')
            })
        
        df = pd.DataFrame(data)
        
        # Estadísticas descriptivas
        stats = {
            'total_participantes': len(df),
            'sd3_promedio': {
                'mach': float(df['mach'].mean()),
                'narc': float(df['narc'].mean()),
                'psych': float(df['psych'].mean())
            },
            'sd3_desviacion': {
                'mach': float(df['mach'].std()),
                'narc': float(df['narc'].std()),
                'psych': float(df['psych'].std())
            },
            'distribucion': {
                'emociones': df['emocion'].value_counts().to_dict(),
                'generos': df['genero'].value_counts().to_dict() if 'genero' in df.columns else {},
                'paises': df['pais'].value_counts().to_dict() if 'pais' in df.columns else {}
            },
            'correlaciones_internas': {
                'mach_narc': float(df['mach'].corr(df['narc'])) if len(df) > 1 else 0,
                'mach_psych': float(df['mach'].corr(df['psych'])) if len(df) > 1 else 0,
                'narc_psych': float(df['narc'].corr(df['psych'])) if len(df) > 1 else 0
            }
        }
        
        # Análisis por grupos
        if 'genero' in df.columns and len(df['genero'].unique()) > 1:
            stats['analisis_por_genero'] = {}
            for genero in df['genero'].unique():
                grupo = df[df['genero'] == genero]
                stats['analisis_por_genero'][genero] = {
                    'n': len(grupo),
                    'mach_promedio': float(grupo['mach'].mean()),
                    'narc_promedio': float(grupo['narc'].mean()),
                    'psych_promedio': float(grupo['psych'].mean())
                }
        
        return {
            "status": "success",
            "estadisticas": stats,
            "muestra": df.head(10).to_dict('records')  # Pequeña muestra
        }
        
    except Exception as e:
        logger.error(f"❌ Error en estadísticas descriptivas: {e}")
        raise HTTPException(status_code=500, detail=f"Error en estadísticas: {str(e)}")

# Para ejecutar en Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
