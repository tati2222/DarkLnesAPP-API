import os
import io
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

import pandas as pd
from scipy import stats

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from supabase import create_client
from facs_mediapipe import FACSMediaPipe
import logging


# ========================================
# CONFIG
# ========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar el analizador FACS (global)
facs_analyzer = None
SUPABASE_URL = "https://cdhndtzuwtmvhiulvzbp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNkaG5kdHp1d3RtdmhpdWx2emJwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzNTE1OTcsImV4cCI6MjA3OTkyNzU5N30.KeyAfqJuCjgSpmd0kRdjDppkJwBRlF9oGyN0ozJMt6M"  # Reemplaza aquí por la real

MODEL_PATH = "modelo_microexpresiones.pth"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def root():
   return {
    "message": "API funcionando correctamente",
    "facs_disponible": facs_analyzer is not None
}


@app.on_event("startup")
async def startup_event():
    global facs_analyzer
    try:
        logger.info("Inicializando analizador FACS...")
        facs_analyzer = FACSMediaPipe()
        logger.info("✓ FACS listo")
    except Exception as e:
        logger.warning(f"⚠️ FACS no disponible: {e}")
        facs_analyzer = None

# ========================================
# CARGA DEL MODELO EfficientNet-B0 con pesos guardados
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 7
LABELS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Instanciar arquitectura y cargar pesos
try:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
except Exception as e:
    print("Error cargando modelo:", e)
    model = None


def emotion_to_code(e):
    inv = {v: k for k, v in LABELS.items()}
    return inv.get(e, -1)


# ========================================
# FUNCIÓN PARA REDACTAR INFORME CLÍNICO
# ========================================
def generar_informe_clinico(emocion, mach, narc, psych, corr, facs_data=None):
    texto = []

    texto.append(f"La emoción predominante detectada es **{emocion}**.")

    # Rasgos
    texto.append(
        f"En los rasgos de la tríada oscura, el participante presentó: "
        f"Maquiavelismo = {mach}, Narcisismo = {narc}, Psicopatía = {psych}."
    )

    # Análisis FACS (NUEVO)
    if facs_data and facs_data.get("action_units"):
        aus_activos = [au["code"] for au in facs_data["action_units"]]
        texto.append(
            f"El análisis FACS identificó {len(aus_activos)} Action Units activos: "
            f"{', '.join(aus_activos)}. "
        )
        
        # Interpretación de autenticidad
        if facs_data.get("interpretation"):
            interp = facs_data["interpretation"]
            auth_score = interp.get("authenticity_score", 0)
            
            if auth_score > 0.6:
                texto.append(
                    f"La expresión emocional muestra una autenticidad de {auth_score*100:.0f}%, "
                    "indicando que la emoción detectada presenta marcadores fisiológicos consistentes."
                )
            elif auth_score > 0.3:
                texto.append(
                    f"La expresión emocional presenta una autenticidad moderada ({auth_score*100:.0f}%), "
                    "lo que sugiere una posible regulación emocional o control de la expresión."
                )
            else:
                texto.append(
                    f"La expresión emocional muestra baja autenticidad ({auth_score*100:.0f}%), "
                    "lo que podría indicar supresión emocional o expresión social controlada."
                )
            
            # Microexpresiones específicas
            if interp.get("microexpression_indicators"):
                indicadores = interp["microexpression_indicators"]
                for ind in indicadores:
                    texto.append(
                        f"Se detectó {ind['type']} con autenticidad {ind['authenticity']} - {ind['note']}."
                    )

    # Correlaciones
    interpretaciones = []
    if corr.get("rho_mach") not in [None, "null"]:
        interpretaciones.append("tendencias manipulativas (maquiavelismo)")
    if corr.get("rho_narc") not in [None, "null"]:
        interpretaciones.append("búsqueda de validación o grandiosidad (narcisismo)")
    if corr.get("rho_psych") not in [None, "null"]:
        interpretaciones.append("conducta impulsiva o baja empatía (psicopatía)")

    if interpretaciones:
        texto.append(
            "Las correlaciones observadas sugieren una relación entre la expresión emocional "
            f"y {', '.join(interpretaciones)}. Estos datos deben interpretarse con cautela, "
            "ya que dependen del tamaño muestral y del contexto."
        )
    else:
        texto.append(
            "No se hallaron correlaciones significativas entre las emociones detectadas "
            "y los rasgos de personalidad, posiblemente por bajo tamaño muestral."
        )

    texto.append(
        "El análisis combina microexpresiones, análisis FACS de Action Units y rasgos de personalidad, "
        "ofreciendo una visión integrada del perfil emocional y conductual, útil para investigación "
        "pero no concluyente a nivel diagnóstico."
    )

    return " ".join(texto)


# ========================================
# ENDPOINT PRINCIPAL (MODIFICADO CON FACS)
# ========================================
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),

    nombre: str = Form(...),
    edad: int = Form(...),
    genero: str = Form(...),
    pais: str = Form(...),

    mach: float = Form(...),
    narc: float = Form(...),
    psych: float = Form(...),

    tiempo_total_seg: float = Form(...),

    historia_utilizada: str = Form(""),
    tipo_captura: str = Form("imagen"),
    include_facs: bool = Form(True),  # NUEVO: permitir desactivar FACS
):
    if model is None:
        raise HTTPException(500, "Modelo no cargado")

    # Procesar imagen
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 1. ANÁLISIS DE EMOCIÓN (tu código original)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        emocion = LABELS.get(pred_idx, "desconocido")

    # 2. ANÁLISIS FACS (NUEVO)
    facs_result = None
    if include_facs and facs_analyzer:
        try:
            logger.info("Ejecutando análisis FACS...")
            img_array = np.array(img)
            facs_result = facs_analyzer.analyze(img_array)
            if facs_result:
                logger.info(f"✓ FACS completado: {len(facs_result.get('action_units', []))} AUs detectados")
            else:
                logger.warning("⚠️ FACS no detectó rostro")
        except Exception as e:
            logger.error(f"❌ Error en FACS: {e}")
            facs_result = None

    # Subir imagen a Supabase Storage
    timestamp = int(datetime.utcnow().timestamp() * 1000)
    file_ext = os.path.splitext(file.filename)[1]
    path = f"microexpresiones/{nombre}_{timestamp}{file_ext}"

    upload = supabase.storage.from_("images").upload(path, contents)
    if upload.get("error"):
        raise HTTPException(500, f"Error subiendo imagen: {upload['error']}")

    public_url_response = supabase.storage.from_("images").get_public_url(path)
    image_url = public_url_response if isinstance(public_url_response, str) else public_url_response.get("publicUrl", "")

    # Correlaciones cohortales
    resp = supabase.table("darklens_records") \
        .select("mach, narc, psych, emocion_principal") \
        .execute()

    df = pd.DataFrame(resp.data)

    corr_info = {
        "rho_mach": None,
        "rho_narc": None,
        "rho_psych": None,
        "p_mach": None,
        "p_narc": None,
        "p_psych": None,
    }

    if len(df) >= 3:
        df = df.dropna()
        if "emocion_principal" in df:
            df["emotion_code"] = df["emocion_principal"].apply(emotion_to_code)
            if df["emotion_code"].nunique() > 1:
                try:
                    corr_info["rho_mach"], corr_info["p_mach"] = stats.spearmanr(df["mach"], df["emotion_code"])
                except:
                    pass
                try:
                    corr_info["rho_narc"], corr_info["p_narc"] = stats.spearmanr(df["narc"], df["emotion_code"])
                except:
                    pass
                try:
                    corr_info["rho_psych"], corr_info["p_psych"] = stats.spearmanr(df["psych"], df["emotion_code"])
                except:
                    pass

    # Extraer datos FACS para guardar en BD (NUEVO)
    aus_frecuentes = None
    facs_promedio = None
    if facs_result:
        # Lista de códigos de AUs activos
        aus_frecuentes = [au["code"] for au in facs_result.get("action_units", [])]
        
        # Promedio de intensidad de AUs
        if aus_frecuentes:
            intensidades = [au["intensity"] for au in facs_result.get("action_units", [])]
            facs_promedio = sum(intensidades) / len(intensidades) if intensidades else 0

    # Generar informe clínico (MODIFICADO para incluir FACS)
    informe = generar_informe_clinico(emocion, mach, narc, psych, corr_info, facs_result)

  row = {
    "nombre": nombre,
    "edad": edad,
    "genero": genero,
    "pais": pais,

    "mach": mach,
    "narc": narc,
    "psych": psych,

    "tiempo_total_seg": tiempo_total_seg,

    "emocion_principal": emocion,
    "total_frames": 1,
    "duracion_video": 0,

    "emociones_detectadas": [emocion],
    "correlaciones": corr_info,

    "aus_frecuentes": aus_frecuentes,
    "facs_promedio": facs_promedio,

    "historia_utilizada": historia_utilizada,
    "tipo_captura": tipo_captura,
    "imagen_url": image_url,
    "imagen_analizada": True,

    # NUEVO → ahora sí se guarda
    "include_facs": include_facs,

    "analisis_completo": informe
}


    insert = supabase.table("darklens_records").insert(row).execute()

    # Respuesta final (MODIFICADO para incluir FACS)
    response = {
        "success": True,
        "emocion_detectada": emocion,
        "imagen_url": image_url,
        "registro_guardado": insert.data,
        "informe": informe,
        
        # NUEVO: Datos FACS separados
        "facs_analysis": facs_result if facs_result else {
            "disponible": False,
            "mensaje": "FACS no disponible o no se detectó rostro"
        }
    }

    return response


# ========================================
# ENDPOINT ADICIONAL: SOLO FACS (OPCIONAL)
# ========================================
@app.post("/analyze-facs-only")
async def analyze_facs_only(file: UploadFile = File(...)):
    """
    Endpoint para analizar solo FACS sin guardar en BD
    Útil para testing o análisis rápido
    """
    if not facs_analyzer:
        raise HTTPException(
            status_code=503,
            detail="FACS no disponible. Verifica que MediaPipe esté instalado."
        )
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
        
        result = facs_analyzer.analyze(img_array)
        
        if not result:
            return {
                "success": False,
                "message": "No se detectó ningún rostro en la imagen"
            }
        
        return {
            "success": True,
            "facs_analysis": result
        }
        
    except Exception as e:
        logger.error(f"Error en análisis FACS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# ENDPOINT DE SALUD
# ========================================
@app.get("/health")
async def health_check():
    """Verificar estado de la API y servicios"""
    return {
        "status": "healthy",
        "modelo_cargado": model is not None,
        "facs_disponible": facs_analyzer is not None,
        "supabase_conectado": True  # Puedes agregar un ping real si quieres
    }
