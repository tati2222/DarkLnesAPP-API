# api.py — versión alineada EXACTAMENTE con tu tabla darklens_records
import os
import io
import json
import traceback
from datetime import datetime

import torch
from torchvision import transforms
from PIL import Image

import numpy as np
import pandas as pd
from scipy import stats

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from supabase import create_client
# ==========================
# CONFIG
# ==========================

SUPABASE_URL = "https://cdhndtzuwtmvhiulvzbp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNkaG5kdHp1d3RtdmhpdWx2emJwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzNTE1OTcsImV4cCI6MjA3OTkyNzU5N30.KeyAfqJuCjgSpmd0kRdjDppkJwBRlF9oGyN0ozJMt6M"

MODEL_PATH = "microexp_retrained_FER2013.pth"   # nombre exacto de tu modelo

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# CARGA DEL MODELO
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
except Exception as e:
    print("Error cargando modelo:", e)
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

# ==========================
# ENDPOINT PRINCIPAL
# ==========================

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
    """
    Recibe:
    - Imagen
    - Datos SD3
    - Datos demográficos
    - Tiempo total
    GUARDADO EN darklens_records
    """

    if model is None:
        raise HTTPException(500, "Modelo no cargado")

    try:
        # =========================================
        # 1) Procesar imagen
        # =========================================
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            pred = torch.argmax(out, dim=1).cpu().numpy()[0]
            emocion = LABELS.get(int(pred), "desconocido")

        # =========================================
        # 2) Subir imagen a Supabase Storage
        # =========================================
        timestamp = int(datetime.utcnow().timestamp() * 1000)
        file_ext = os.path.splitext(file.filename)[1]
        path = f"microexpresiones/{nombre}_{timestamp}{file_ext}"

        upload = supabase.storage.from_("images").upload(path, contents)

        if upload.get("error"):
            raise HTTPException(500, f"Error subiendo imagen: {upload['error']}")

        image_url = supabase.storage.from_("images").get_public_url(path).get("publicURL")

        # =========================================
        # 3) Calcular correlaciones cohortales
        # =========================================
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
            "interpretacion": "Aún no hay suficientes datos."
        }

        if len(df) >= 3:
            df = df.dropna()

            if "emocion_principal" in df:
                df["emotion_code"] = df["emocion_principal"].apply(emotion_to_code)

                if df["emotion_code"].nunique() > 1:

                    # MACH
                    try:
                        rho, p = stats.spearmanr(df["mach"], df["emotion_code"])
                        corr_info["rho_mach"] = rho
                        corr_info["p_mach"] = p
                    except:
                        pass

                    # NARC
                    try:
                        rho, p = stats.spearmanr(df["narc"], df["emotion_code"])
                        corr_info["rho_narc"] = rho
                        corr_info["p_narc"] = p
                    except:
                        pass

                    # PSYCH
                    try:
                        rho, p = stats.spearmanr(df["psych"], df["emotion_code"])
                        corr_info["rho_psych"] = rho
                        corr_info["p_psych"] = p
                    except:
                        pass

                    corr_info["interpretacion"] = "Correlaciones calculadas. Interpretar con cautela."

        # =========================================
        # 4) Guardar registro final en tabla
        # =========================================

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
            "total_frames": 1,  # si después agregás video, esto cambia
            "duracion_video": 0,

            "emociones_detectadas": [emocion],
            "correlaciones": corr_info,

            "aus_frecuentes": [],
            "facs_promedio": {},

            "historia_utilizada": historia_utilizada,
            "tipo_captura": tipo_captura,
            "imagen_analizada": True,

            "analisis_completo": f"La emoción detectada es {emocion}.",
        }

        insert = supabase.table("darklens_records").insert(row).execute()

        return {
            "success": True,
            "emocion_detectada": emocion,
            "imagen_url": image_url,
            "registro_guardado": insert.data,
            "correlaciones":
