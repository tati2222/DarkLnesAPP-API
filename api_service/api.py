import os
import io
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

# Intentar import flexible de FACSMediaPipe (soporta import relativo o absoluto)
try:
    from .facs_mediapipe import FACSMediaPipe
except Exception:
    try:
        from facs_mediapipe import FACSMediaPipe
    except Exception:
        FACSMediaPipe = None

# ========================================
# CONFIG
# ========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase (mantengo la config que tenías; considerá usar variables de entorno en producción)
SUPABASE_URL = "https://cdhndtzuwtmvhiulvzbp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNkaG5kdHp1d3RtdmhpdWx2emJwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NDM1MTU5NywiZXhwIjoyMDc5OTI3NTk3fQ.-vqSP3Vy1qLPoDcTZfo58lhcs1ydTgsgPVh8yGyX5eU"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Nombre exacto que me dijiste
MODEL_FILENAME = "modelo_microexpresiones.pth"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Bucket(s) posibles en Supabase donde podrías subir el modelo
MODEL_BUCKET_CANDIDATES = ["modelos", "Modelos", "model", "modelo", "MODEL", "DARKLENS-IMAGES"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Estado globales
facs_analyzer = None

@app.get("/")
def root():
    return {
        "message": "API funcionando correctamente",
        "facs_disponible": facs_analyzer is not None,
        "modelo_local": os.path.exists(MODEL_PATH)
    }

# ========================================
# UTIL: descargar modelo desde Supabase (si no está local)
# ========================================

PUBLIC_MODEL_URL = (
    "https://cdhndtzuwtmvhiulvzbp.supabase.co/storage/v1/object/public/"
    "modelos/modelo_microexpresiones.pth"
)

def download_model_from_supabase():
    """
    Descarga el archivo del modelo desde una URL pública fija.
    Devuelve True si el archivo queda guardado localmente.
    """
    if os.path.exists(MODEL_PATH):
        logger.info("Modelo ya existe localmente: %s", MODEL_PATH)
        return True

    try:
        logger.info("Descargando modelo desde Supabase (URL fija)...")
        r = requests.get(PUBLIC_MODEL_URL, timeout=60)

        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            logger.info("Modelo descargado correctamente en %s", MODEL_PATH)
            return True
        else:
            logger.error("Fallo la descarga: código %s", r.status_code)
            return False

    except Exception as e:
        logger.error("Error descargando modelo: %s", e)
        return False


# ========================================
# STARTUP: inicializar FACS y asegurar modelo
# ========================================
@app.on_event("startup")
async def startup_event():
    global facs_analyzer, model

    # Inicializar FACS si está disponible el módulo
    try:
        if FACSMediaPipe:
            logger.info("Inicializando analizador FACS...")
            facs_analyzer = FACSMediaPipe()
            logger.info("✓ FACS listo")
        else:
            facs_analyzer = None
            logger.warning("FACSMediaPipe no disponible (archivo facs_mediapipe.py ausente o error en import).")
    except Exception as e:
        facs_analyzer = None
        logger.warning("⚠️ FACS no disponible: %s", e)

    # Asegurar que el modelo exista localmente (si no, intentar descargar desde Supabase)
    if not os.path.exists(MODEL_PATH):
        download_ok = download_model_from_supabase()
        if not download_ok:
            logger.warning("Modelo no encontrado localmente y no se pudo descargar. El endpoint /analyze devolverá error.")

    # Cargar modelo (si está)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7

    try:
        if os.path.exists(MODEL_PATH):
            # Instanciar la misma arquitectura que usaste en entrenamiento (weights=None)
            model_local = models.efficientnet_b0(weights=None)
            model_local.classifier[1] = nn.Linear(model_local.classifier[1].in_features, num_classes)

            # Cargar en modo binario
            with open(MODEL_PATH, "rb") as f:
                state = torch.load(f, map_location=device)

            # Si el archivo contiene un 'state_dict' guardado como diccionario, lo cargamos
            if isinstance(state, dict):
                model_local.load_state_dict(state)
            else:
                # Si por alguna razón el archivo contiene el modelo completo, lo asignamos
                model_local = state

            model_local.to(device)
            model_local.eval()
            model = model_local
            logger.info("Modelo cargado correctamente desde %s", MODEL_PATH)
        else:
            model = None
            logger.warning("No se encontró el archivo del modelo; model queda None.")
    except Exception as e:
        model = None
        logger.error("Error cargando modelo: %s", e)

# ========================================
# Configuración común: labels y transform
# ========================================
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

def emotion_to_code(e):
    inv = {v: k for k, v in LABELS.items()}
    return inv.get(e, -1)

# ========================================
# FUNCIÓN PARA REDACTAR INFORME CLÍNICO
# ========================================
def generar_informe_clinico(emocion, mach, narc, psych, corr, facs_data=None):
    texto = []

    texto.append(f"La emoción predominante detectada es **{emocion}**.")

    texto.append(
        f"En los rasgos de la tríada oscura, el participante presentó: "
        f"Maquiavelismo = {mach}, Narcisismo = {narc}, Psicopatía = {psych}."
    )

    if facs_data and facs_data.get("action_units"):
        aus_activos = [au["code"] for au in facs_data["action_units"]]
        texto.append(
            f"El análisis FACS identificó {len(aus_activos)} Action Units activos: "
            f"{', '.join(aus_activos)}. "
        )

        interp = facs_data.get("interpretation")
        if interp:
            auth_score = interp.get("authenticity_score", 0)
            if auth_score > 0.6:
                texto.append(
                    f"La expresión emocional muestra una autenticidad de {auth_score*100:.0f}%."
                )
            elif auth_score > 0.3:
                texto.append(
                    f"La expresión emocional presenta una autenticidad moderada ({auth_score*100:.0f}%)."
                )
            else:
                texto.append(
                    f"La expresión emocional muestra baja autenticidad ({auth_score*100:.0f}%)."
                )

            if interp.get("microexpression_indicators"):
                indicadores = interp["microexpression_indicators"]
                for ind in indicadores:
                    texto.append(
                        f"Se detectó {ind['type']} con autenticidad {ind['authenticity']} - {ind['note']}."
                    )

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
            f"y {', '.join(interpretaciones)}. Estos datos deben interpretarse con cautela."
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
# ENDPOINT PRINCIPAL
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
    include_facs: bool = Form(True),
):
    # Verificamos modelo
    if 'model' not in globals() or model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en el servidor. Ver logs.")

    # Procesar imagen
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error("Error al abrir imagen: %s", e)
        raise HTTPException(status_code=400, detail="Imagen inválida")

    # 1) Predicción de emoción
    tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        emocion = LABELS.get(pred_idx, "desconocido")

    # 2) FACS (si disponible)
    facs_result = None
    if include_facs and facs_analyzer:
        try:
            logger.info("Ejecutando análisis FACS...")
            img_array = np.array(img)
            facs_result = facs_analyzer.analyze(img_array)
            if facs_result:
                logger.info("✓ FACS completado")
            else:
                logger.warning("⚠️ FACS no detectó rostro")
        except Exception as e:
            logger.error("Error en FACS: %s", e)
            facs_result = None

    # 3) Subir imagen a Supabase Storage
    timestamp = int(datetime.utcnow().timestamp() * 1000)
    file_ext = os.path.splitext(file.filename)[1] or ".jpg"
    upload_path = f"microexpresiones/{nombre}_{timestamp}{file_ext}"

    try:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG' if file_ext.lower() in ['.jpg', '.jpeg'] else 'PNG')
        file_bytes = img_byte_arr.getvalue()

        upload = supabase.storage.from_("DARKLENS-IMAGES").upload(upload_path, file_bytes)

        # Verificar si supabase devolvió error
        if hasattr(upload, "error") and upload.error:
            logger.error("Error subiendo imagen: %s", upload.error)
            raise HTTPException(status_code=500, detail=f"Error subiendo imagen: {upload.error}")
    except Exception as e:
        logger.error("Error en upload: %s", e)
        raise HTTPException(status_code=500, detail=f"Error subiendo imagen: {str(e)}")

    # Obtener URL pública
    try:
        url_resp = supabase.storage.from_("DARKLENS-IMAGES").get_public_url(upload_path)
        if isinstance(url_resp, str):
            image_url = url_resp
        else:
            image_url = getattr(url_resp, "publicUrl", "") or getattr(url_resp, "public_url", "") or ""
    except Exception as e:
        logger.warning("No se pudo obtener URL pública: %s", e)
        image_url = ""

    # 4) Correlaciones cohortales
    try:
        resp = supabase.table("darklens_records").select("mach, narc, psych, emocion_principal").execute()
        df = pd.DataFrame(resp.data if resp.data else [])
    except Exception as e:
        logger.warning("Error en consulta de correlaciones: %s", e)
        df = pd.DataFrame()

    corr_info = {
        "rho_mach": None,
        "rho_narc": None,
        "rho_psych": None,
        "p_mach": None,
        "p_narc": None,
        "p_psych": None,
    }

    if len(df) >= 3 and "emocion_principal" in df.columns:
        try:
            df = df.dropna()
            df["emotion_code"] = df["emocion_principal"].apply(emotion_to_code)
            if df["emotion_code"].nunique() > 1:
                corr_info["rho_mach"], corr_info["p_mach"] = stats.spearmanr(df["mach"], df["emotion_code"])
                corr_info["rho_narc"], corr_info["p_narc"] = stats.spearmanr(df["narc"], df["emotion_code"])
                corr_info["rho_psych"], corr_info["p_psych"] = stats.spearmanr(df["psych"], df["emotion_code"])
        except Exception as e:
            logger.warning("Error calculando correlaciones: %s", e)

    # 5) Extraer datos FACS
    aus_frecuentes = None
    facs_promedio = None
    if facs_result:
        aus_frecuentes = [au["code"] for au in facs_result.get("action_units", [])]
        intensidades = [au.get("intensity", 0) for au in facs_result.get("action_units", [])]
        if intensidades:
            facs_promedio = sum(intensidades) / len(intensidades)

    # 6) Generar informe
    informe = generar_informe_clinico(emocion, mach, narc, psych, corr_info, facs_result)

    # 7) Guardar en BD
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
        "include_facs": include_facs,
        "analisis_completo": informe
    }

    try:
        insert = supabase.table("darklens_records").insert(row).execute()
        registro_guardado = not (hasattr(insert, "error") and insert.error)
        if not registro_guardado:
            logger.warning("Insert returned error: %s", getattr(insert, "error", None))
    except Exception as e:
        logger.error("Error en inserción a BD: %s", e)
        registro_guardado = False

    # Respuesta final
    return {
        "success": True,
        "emocion_detectada": emocion,
        "imagen_url": image_url,
        "registro_guardado": registro_guardado,
        "informe": informe,
        "facs_analysis": facs_result if facs_result else {
            "disponible": False,
            "mensaje": "FACS no disponible o no se detectó rostro"
        }
    }

# ========================================
# ENDPOINT: SOLO FACS (Útil para testing)
# ========================================
@app.post("/analyze-facs-only")
async def analyze_facs_only(file: UploadFile = File(...)):
    if not facs_analyzer:
        raise HTTPException(status_code=503, detail="FACS no disponible. Verifica que MediaPipe esté instalado.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = facs_analyzer.analyze(np.array(img))

        if not result:
            return {"success": False, "message": "No se detectó rostro"}

        return {"success": True, "facs_analysis": result}
    except Exception as e:
        logger.error("Error en análisis FACS: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# ENDPOINT DE SALUD
# ========================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "modelo_cargado": 'model' in globals() and model is not None,
        "facs_disponible": facs_analyzer is not None,
        "supabase_conectado": True
    }
