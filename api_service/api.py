import os
import io
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models
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
SUPABASE_KEY = "YOUR_SUPABASE_KEY"  # Reemplaza aquí por la real

MODEL_PATH = "best_microexp_model.pth"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

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

    # Procesar imagen
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        emocion = LABELS.get(pred_idx, "desconocido")

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

    # Generar informe clínico
    informe = generar_informe_clinico(emocion, mach, narc, psych, corr_info)

    # Guardar registro final
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
        "imagen_url": image_url, 
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
