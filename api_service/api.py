# api.py — versión final
import os
import io
import json
from datetime import datetime

import torch
from torchvision import transforms
from PIL import Image

import pandas as pd
from scipy import stats

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from supabase import create_client

# ========================================
# CONFIG
# ========================================
SUPABASE_URL = "https://cdhndtzuwtmvhiulvzbp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNkaG5kdHp1d3RtdmhpdWx2emJwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzNTE1OTcsImV4cCI6MjA3OTkyNzU5N30.KeyAfqJuCjgSpmd0kRdjDppkJwBRlF9oGyN0ozJMt6M"

MODEL_PATH = "microexp_retrained_FER2013.pth"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ========================================
# CARGA DEL MODELO
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
except:
    model = None

LABELS = {
    0: "neutral",
    1: "felicidad",
    2: "tristeza",
    3: "sorpresa",
    4: "miedo",
    5: "asco",
    6: "enojo"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def emotion_to_code(e):
    inv = {v: k for k, v in LABELS.items()}
    return inv.get(e, -1)


# ========================================
# FUNCIÓN PARA REDACTAR INFORME CLÍNICO
# ========================================
def generar_informe_clinico(emocion, mach, narc, psych, corr):
    texto = []

    texto.append(f"La emoción predominante detectada es **{emocion}**.")

    # Rasgos
    texto.append(
        f"En los rasgos de la tríada oscura, el participante presentó: "
        f"Maquiavelismo = {mach}, Narcisismo = {narc}, Psicopatía = {psych}."
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
        "El análisis combina microexpresiones y rasgos de personalidad, ofreciendo una "
        "visión integrada del perfil emocional y conductual, útil para investigación "
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
):
    if model is None:
        raise HTTPException(500, "Modelo no cargado")

    # ================================
    # Procesar imagen
    # ================================
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        pred = torch.argmax(out, dim=1).cpu().numpy()[0]
        emocion = LABELS.get(int(pred), "desconocido")

    # ================================
    # Subir imagen a Supabase Storage
    # ================================
    timestamp = int(datetime.utcnow().timestamp() * 1000)
    file_ext = os.path.splitext(file.filename)[1]
    path = f"microexpresiones/{nombre}_{timestamp}{file_ext}"

    upload = supabase.storage.from_("images").upload(path, contents)
    if upload.get("error"):
        raise HTTPException(500, f"Error subiendo imagen: {upload['error']}")

    image_url = supabase.storage.from_("images").get_public_url(path).get("publicURL")

    # ================================
    # Correlaciones cohortales
    # ================================
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

    # ================================
    # GENERAR INFORME CLÍNICO
    # ================================
    informe = generar_informe_clinico(emocion, mach, narc, psych, corr_info)

    # ================================
    # Guardar registro final
    # ================================
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

        "aus_frecuentes": None,
        "facs_promedio": None,

        "historia_utilizada": historia_utilizada,
        "tipo_captura": tipo_captura,
        "imagen_analizada": True,

        "analisis_completo": informe
    }

    insert = supabase.table("darklens_records").insert(row).execute()

    return {
        "success": True,
        "emocion_detectada": emocion,
        "imagen_url": image_url,
        "registro_guardado": insert.data,
        "informe": informe
    }
