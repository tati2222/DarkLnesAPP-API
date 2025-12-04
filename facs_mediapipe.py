# facs_analyzer.py - Módulo separado para FACS/AUs
from feat import Detector
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FACSAnalyzer:
    """Analizador de FACS/AUs independiente"""
    
    def __init__(self):
        self.detector = None
        self.load_models()
    
    def load_models(self):
        """Cargar modelos de PyFeat"""
        try:
            logger.info("Cargando modelos PyFeat para FACS...")
            self.detector = Detector(
                face_model="retinaface",
                landmark_model="mobilefacenet",
                au_model="xgb",
                emotion_model="resmasknet"
            )
            logger.info("Modelos FACS cargados")
        except Exception as e:
            logger.error(f"Error cargando FACS: {e}")
            self.detector = None
    
    def analyze(self, image_array: np.ndarray) -> Optional[Dict]:
        """
        Analizar AUs en una imagen
        Retorna None si no hay detector o falla
        """
        if not self.detector:
            return None
        
        try:
            result = self.detector.detect_image(image_array)
            
            if len(result) == 0:
                return None
            
            face_data = result.iloc[0]
            
            # Extraer Action Units
            au_columns = [col for col in result.columns if col.startswith('AU')]
            aus_raw = {col: float(face_data[col]) for col in au_columns}
            
            # Procesar AUs activos
            active_aus = self._process_aus(aus_raw)
            
            # Extraer emociones de PyFeat
            emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
            emotions = {col: float(face_data[col]) for col in emotion_cols if col in result.columns}
            
            # Interpretación
            interpretation = self._interpret_aus(active_aus, emotions)
            
            return {
                "action_units": active_aus,
                "emotions_pyfeat": emotions,
                "interpretation": interpretation,
                "confidence": float(face_data.get('FaceScore', 0))
            }
            
        except Exception as e:
            logger.error(f"Error en análisis FACS: {e}")
            return None
    
    def _process_aus(self, aus_dict: Dict) -> List[Dict]:
        """Procesar Action Units activos"""
        au_descriptions = {
            "AU01": "Elevación párpado interior (sorpresa, miedo)",
            "AU02": "Elevación párpado exterior (sorpresa)",
            "AU04": "Ceño fruncido (concentración, ira)",
            "AU05": "Elevación párpado superior (atención)",
            "AU06": "Mejillas levantadas (sonrisa genuina)",
            "AU07": "Párpados tensos (intensidad)",
            "AU09": "Arruga nariz (disgusto)",
            "AU10": "Labio superior levantado (disgusto)",
            "AU12": "Comisuras labios arriba (sonrisa)",
            "AU14": "Hoyuelos (sonrisa intensa)",
            "AU15": "Comisuras labios abajo (tristeza)",
            "AU17": "Barbilla levantada (duda)",
            "AU20": "Estiramiento labios (miedo)",
            "AU23": "Labios apretados (tensión)",
            "AU24": "Labios presionados (supresión)",
            "AU25": "Labios separados (relajación)",
            "AU26": "Mandíbula caída (sorpresa)",
        }
        
        active = []
        for au, value in aus_dict.items():
            if au.startswith("AU") and value > 0.5:
                active.append({
                    "code": au,
                    "intensity": value,
                    "description": au_descriptions.get(au, "N/A")
                })
        
        return sorted(active, key=lambda x: x["intensity"], reverse=True)
    
    def _interpret_aus(self, aus: List[Dict], emotions: Dict) -> Dict:
        """Interpretar combinaciones de AUs"""
        au_codes = [au["code"] for au in aus]
        indicators = []
        authenticity = 0.0
        
        # Sonrisa Duchenne (genuina)
        if "AU06" in au_codes and "AU12" in au_codes:
            indicators.append({
                "type": "Sonrisa de Duchenne",
                "authenticity": "Alta",
                "note": "Sonrisa genuina"
            })
            authenticity += 0.3
        
        # Sonrisa social (falsa)
        elif "AU12" in au_codes and "AU06" not in au_codes:
            indicators.append({
                "type": "Sonrisa social",
                "authenticity": "Baja",
                "note": "Posiblemente cortés o forzada"
            })
        
        # Disgusto
        if "AU09" in au_codes or "AU10" in au_codes:
            indicators.append({
                "type": "Disgusto",
                "authenticity": "Media-Alta",
                "note": "Señales de desagrado"
            })
            authenticity += 0.2
        
        # Miedo/Sorpresa
        fear_aus = sum(1 for au in ["AU01", "AU02", "AU05"] if au in au_codes)
        if fear_aus >= 2:
            indicators.append({
                "type": "Miedo/Sorpresa",
                "authenticity": "Media",
                "note": "Expresión de alarma"
            })
            authenticity += 0.2
        
        # Tristeza
        if "AU15" in au_codes or "AU17" in au_codes:
            indicators.append({
                "type": "Tristeza",
                "authenticity": "Media",
                "note": "Indicadores de abatimiento"
            })
            authenticity += 0.2
        
        primary_emotion = max(emotions, key=emotions.get) if emotions else "neutral"
        
        return {
            "primary_emotion": primary_emotion,
            "confidence": float(max(emotions.values())) if emotions else 0.0,
            "microexpression_indicators": indicators,
            "authenticity_score": min(1.0, authenticity)
        }


# main_api.py - Tu API principal MODIFICADA
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import logging

# Importar tu análisis de microexpresiones existente
# from tu_modulo_microexpresiones import analizar_microexpresiones

# Importar el nuevo módulo FACS
from facs_analyzer import FACSAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Microexpression Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar analizador FACS (opcional)
facs_analyzer = None

@app.on_event("startup")
async def startup():
    global facs_analyzer
    try:
        facs_analyzer = FACSAnalyzer()
    except Exception as e:
        logger.warning(f"FACS no disponible: {e}")

@app.get("/")
async def root():
    return {
        "message": "Microexpression Analysis API",
        "endpoints": {
            "/analyze": "POST - Análisis completo (microexp + FACS opcional)",
            "/analyze-micro": "POST - Solo microexpresiones",
            "/analyze-facs": "POST - Solo FACS/AUs",
            "/health": "GET - Estado"
        },
        "facs_available": facs_analyzer is not None
    }

@app.post("/analyze")
async def analyze_complete(
    file: UploadFile = File(...),
    include_facs: bool = True
):
    """
    Análisis completo: microexpresiones + FACS (opcional)
    """
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        # 1. TU ANÁLISIS DE MICROEXPRESIONES EXISTENTE
        # Aquí va tu código actual
        microexpression_result = {
            "detected": True,
            # ... tus resultados actuales
            "emotions": {},
            "patterns": []
        }
        
        # 2. ANÁLISIS FACS (OPCIONAL Y SEPARADO)
        facs_result = None
        if include_facs and facs_analyzer:
            facs_result = facs_analyzer.analyze(img_array)
        
        # 3. RESPUESTA COMBINADA PERO SEPARADA
        response = {
            "success": True,
            "microexpression_analysis": microexpression_result,
            "facs_analysis": facs_result,  # Puede ser None
            "combined_insights": None
        }
        
        # 4. INSIGHTS COMBINADOS (opcional)
        if facs_result:
            response["combined_insights"] = {
                "correlation": "Análisis combinado aquí",
                "consistency": "Comparar resultados"
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-micro")
async def analyze_microexpressions_only(file: UploadFile = File(...)):
    """
    Solo tu análisis de microexpresiones original
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        # TU CÓDIGO EXISTENTE AQUÍ
        result = {
            "success": True,
            # ... tu análisis
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-facs")
async def analyze_facs_only(file: UploadFile = File(...)):
    """
    Solo análisis FACS/AUs
    """
    if not facs_analyzer:
        raise HTTPException(
            status_code=503,
            detail="FACS no disponible. Instalar py-feat"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        result = facs_analyzer.analyze(img_array)
        
        if not result:
            return {
                "success": False,
                "message": "No se detectaron rostros"
            }
        
        return {
            "success": True,
            "facs_analysis": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "facs_enabled": facs_analyzer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
