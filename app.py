# app.py - DarkLens (Streamlit)
# Versión completa con FAQ, explicación FAC, guardado local y opción Google Sheets.
# Requisitos (requirements.txt): streamlit, torch, torchvision, pillow, pandas, gspread, google-auth
# (Instala gspread y google-auth solo si vas a usar la integración con Google Sheets)

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import pandas as pd
import numpy as np
import os
import io
import datetime
import json

# Optional imports for Google Sheets (import only if enabled)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GS_AVAILABLE = True
except Exception:
    GS_AVAILABLE = False

# --------------------------
# CONFIGURACIÓN PÁGINA
# --------------------------
st.set_page_config(page_title="DarkLens", page_icon="🟣", layout="wide")

# CSS personalizado
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at center, #3a0066, #14001f);
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #ffffff !important;
    }
    .stButton>button {
        background: #6a0dad !important;
        color: white !important;
        border-radius: 8px !important;
        width: 100%;
        padding: 0.5rem;
    }
    .conclusion-box {
        background: rgba(168, 85, 247, 0.12);
        border-left: 4px solid #a855f7;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .emotion-dominant {
        font-size: 1.25rem;
        font-weight: bold;
        color: #a855f7;
    }
    .warning-box {
        background: rgba(236, 72, 153, 0.12);
        border-left: 4px solid #ec4899;
        padding: 0.9rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-box {
        background: rgba(255,255,255,0.03);
        padding: 0.9rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Rutas / Configuraciones
# --------------------------

# PONER AQUÍ tu modelo si ya lo subiste a /mnt/data o a la raíz del repo
# Ejemplos:
# MODEL_PATH = "/content/microexp_retrained_FER2013.pth"
# MODEL_PATH = "/mnt/data/emotion_model_finetuned (1).keras"  # <-- si tienes el .keras
MODEL_PATH = "microexp_retrained_FER2013.pth"

# Opcional: ruta al JSON de servicio de Google para subir a Google Sheets
# Si no usás Sheets deja vacío o comentá la línea.
GSHEET_CREDENTIALS_PATH = ""  # p.ej. "/path/to/service_account.json"
GSHEET_NAME = "DarkLens_Results"  # nombre de la hoja (crear o se crea)

# CSV local donde se guardan resultados si quieres
LOCAL_RESULTS_CSV = "darklens_results.csv"

# --------------------------
# UTIL: FAC (FACS) mapping (descriptivo)
# --------------------------
# Diccionario simple que vincula emoción -> AUs típicos y descripción en lenguaje humano.
FAC_MAPPING = {
    "Alegría": {
        "AUs": ["AU6 (Orbicularis oculi)", "AU12 (Zygomaticus major)"],
        "descripcion": "Sonrisa genuina: elevación de mejillas y arrugamiento alrededor de los ojos (patas de gallo)."
    },
    "Tristeza": {
        "AUs": ["AU1+4 (Frontalis/Depressor)", "AU15 (Depressor anguli oris)"],
        "descripcion": "Comisura de los labios hacia abajo, párpados pesados y cejas arqueadas en el centro."
    },
    "Enojo": {
        "AUs": ["AU4 (Brow lowerer)", "AU7 (Lid tightener)", "AU23 (Lip tightener)"],
        "descripcion": "Ceño fruncido, tensión en la mandíbula y mirada fija/intensa."
    },
    "Sorpresa": {
        "AUs": ["AU1+2 (Inner/Outer brow raiser)", "AU5 (Upper lid raiser)", "AU26 (Jaw drop)"],
        "descripcion": "Cejas elevadas, ojos muy abiertos y boca ligeramente entreabierta."
    },
    "Miedo": {
        "AUs": ["AU1+2", "AU5", "AU20 (Lip stretcher)"],
        "descripcion": "Cejas tensas, ojos abiertos y labios tensos; expresión de alerta y retirada."
    },
    "Disgusto": {
        "AUs": ["AU9 (Nose wrinkler)", "AU10 (Upper lip raiser)"],
        "descripcion": "Arrugamiento de la nariz y elevación del labio superior, como rechazo."
    },
    "Neutral": {
        "AUs": [],
        "descripcion": "Ausencia de configuraciones faciales marcadas; rostro relajado o controlado."
    }
}

# --------------------------
# MODELO: wrapper general (EfficientNet-B0)
# --------------------------
class MicroExpNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.net = efficientnet_b0(weights=None)
        in_features = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.net(x)

# --------------------------
# CARGA MODELO ROBUSTA
# --------------------------
@st.cache_resource
def cargar_modelo_robusto(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicroExpNet(num_classes=7)
    model_loaded = False
    error_message = None

    # Intentar cargar modelos .pth / .pt
    if os.path.isfile(model_path):
        lower = model_path.lower()
        try:
            if lower.endswith(".pth") or lower.endswith(".pt"):
                state = torch.load(model_path, map_location=device)
                # state puede ser: state_dict directo, o checkpoint con 'model_state_dict' u otros prefijos
                if isinstance(state, dict):
                    # detectar claves conocidas
                    # opciones: state_dict directo (key names empiezan por 'net.' o 'model.' o sin prefijo)
                    keys = list(state.keys())
                    # si está guardado como checkpoint con 'model_state_dict' o 'state_dict'
                    if 'model_state_dict' in state:
                        sd = state['model_state_dict']
                        model.load_state_dict(sd, strict=False)
                        model_loaded = True
                    elif 'state_dict' in state:
                        sd = state['state_dict']
                        model.load_state_dict(sd, strict=False)
                        model_loaded = True
                    else:
                        # heurísticos para limpiar posibles prefijos
                        first_key = keys[0]
                        sd = state
                        if first_key.startswith('model.model.') or first_key.startswith('net.'):
                            # eliminar un prefijo de más si existe
                            new_sd = {}
                            for k, v in state.items():
                                new_key = k
                                # quitar "model." una sola vez si aparece al inicio
                                if new_key.startswith('model.'):
                                    new_key = new_key.replace('model.', '', 1)
                                if new_key.startswith('net.'):
                                    new_key = new_key.replace('net.', '', 1)
                                new_sd[new_key] = v
                            try:
                                model.load_state_dict(new_sd, strict=False)
                                model_loaded = True
                            except Exception:
                                # fallback: intentar cargar tal cual
                                try:
                                    model.load_state_dict(state, strict=False)
                                    model_loaded = True
                                except Exception as e:
                                    error_message = str(e)
                        else:
                            # intento directo
                            try:
                                model.load_state_dict(state, strict=False)
                                model_loaded = True
                            except Exception as e:
                                # tal vez el checkpoint tiene pesos en subclave 'model'
                                # intentar acomodar prefijo 'model.' si las keys del modelo comienzan con 'model.'
                                try:
                                    # buscar si keys del modelo comienzan por 'model.' (no común aquí)
                                    model.load_state_dict(state, strict=False)
                                    model_loaded = True
                                except Exception as e2:
                                    error_message = str(e2)
                else:
                    error_message = "El archivo .pth cargado no es un dict reconocible."
            elif lower.endswith(".keras") or lower.endswith(".h5"):
                # Intentar cargar keras y convertir a PyTorch no es trivial.
                # Aquí informamos al usuario cómo proceder: preferible convertir a ONNX o re-entrenar.
                error_message = ("Modelo Keras detectado (.keras/.h5). "
                                 "La app espera un .pth de PyTorch. Para usar ese archivo, convértilo a ONNX o guarda un state_dict de PyTorch.")
            else:
                error_message = "Formato desconocido. Use un archivo .pth/.pt o convierta a formato compatible."
        except Exception as e:
            error_message = str(e)
    else:
        error_message = f"No se encontró el archivo de modelo en: {model_path}"

    if not model_loaded:
        raise RuntimeError(f"Error cargando el modelo: {error_message}")

    model.to(device)
    model.eval()
    return model, device

# --------------------------
# TRANSFORMS Y LABELS
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

LABELS = ["Alegría", "Tristeza", "Enojo", "Sorpresa", "Miedo", "Disgusto", "Neutral"]

# --------------------------
# FUNCIONES SD3 (simple mapping experimental)
# --------------------------
def compute_sd3_from_emotions(emotions: dict):
    """
    Función heurística experimental: transforma la distribución de emociones en puntajes SD3.
    Esta función es exploratoria y debe documentarse como tal en el informe.
    """
    maqu = emotions.get("Enojo",0)*0.6 + emotions.get("Disgusto",0)*0.4
    narc = emotions.get("Alegría",0)*0.5 + emotions.get("Neutral",0)*0.5
    psic = emotions.get("Miedo",0)*0.7 + emotions.get("Sorpresa",0)*0.3
    # convertir a escala 0-100
    return {
        "Maquiavelismo": round(maqu*100,2),
        "Narcisismo": round(narc*100,2),
        "Psicopatía": round(psic*100,2)
    }

# --------------------------
# EXPLICACIÓN FAC basada en la predicción
# --------------------------
def explain_fac(emotions: dict):
    # emoción dominante
    dominant = max(emotions.items(), key=lambda x: x[1])
    emo_name, emo_prob = dominant
    mapping = FAC_MAPPING.get(emo_name, {})
    aus = mapping.get("AUs", [])
    desc = mapping.get("descripcion", "Descripción no disponible.")
    # preparar texto explicativo
    texto = f"La microexpresión dominante es **{emo_name}** ({emo_prob*100:.1f}%).\n\n"
    texto += f"{desc}\n\n"
    if aus:
        texto += "Unidades de acción (AUs) implicadas: " + ", ".join(aus) + "."
    else:
        texto += "No se identifican AUs claramente en 'Neutral'."
    return texto, emo_name, emo_prob, aus

# --------------------------
# GUARDAR & SUBIR (Google Sheets opcional)
# --------------------------
def save_result_local(record: dict, csv_path=LOCAL_RESULTS_CSV):
    df = pd.DataFrame([record])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def upload_to_gsheets(record: dict, credentials_path: str, sheet_name: str):
    if not GS_AVAILABLE:
        raise RuntimeError("gspread / google-auth no están instalados en este entorno.")
    if not os.path.isfile(credentials_path):
        raise RuntimeError("No se encontró el archivo de credenciales JSON.")
    creds = Credentials.from_service_account_file(credentials_path, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    gc = gspread.authorize(creds)
    try:
        sh = gc.open(sheet_name)
    except Exception:
        sh = gc.create(sheet_name)
        # compartir la hoja si es necesario (requiere permisos de cuenta)
    worksheet = sh.sheet1
    # escribir encabezados si vacía
    headers = worksheet.row_values(1)
    if not headers:
        worksheet.append_row(list(record.keys()))
    worksheet.append_row(list(record.values()))

# --------------------------
# PREDICCIÓN
# --------------------------
def predict_from_image(img_pil: Image.Image, model, device):
    img = img_pil.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)  # logits
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    # mapear a etiquetas
    emotions = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return emotions

# --------------------------
# CARGAR MODELO AL INICIO (intentar y mostrar info)
# --------------------------
# Intentamos cargar al ejecutar la app
model_load_error = None
model = None
device = None
try:
    model, device = cargar_modelo_robusto(MODEL_PATH)
    st.sidebar.success(f"Modelo cargado desde: {MODEL_PATH} (device: {device})")
except Exception as e:
    model_load_error = str(e)
    st.sidebar.error("No se pudo cargar el modelo. Revisa MODEL_PATH en app.py y el formato del archivo.")
    st.sidebar.write(model_load_error)

# --------------------------
# INTERFAZ PRINCIPAL
# --------------------------
st.markdown("<h1 style='text-align:center;'>🟣 DarkLens — Detector de Microexpresiones</h1>", unsafe_allow_html=True)

# Palabras clave (keywords) para el informe
st.sidebar.markdown("**Palabras clave:** microexpresiones, SD3, Dark Triad, visión computacional, FACS, ética de datos")

# Sección FAQ integrada
with st.expander("❓ FAQ / Preguntas frecuentes (ver antes de usar)"):
    st.markdown("""
    **¿Qué hace DarkLens?**  
    *DarkLens procesa una imagen facial con un modelo pre-entrenado y devuelve la microexpresión predominante (7 clases),
    además de una transformación heurística experimental hacia puntajes SD3. Esto es exploratorio y no diagnóstico.*
    
    **¿Puedo usar esto para diagnosticar a una persona?**  
    No. DarkLens no es una herramienta clínica ni forense. Sus salidas son probabilísticas y deben interpretarse en contexto.
    
    **¿Dónde se guardan las imágenes?**  
    Por defecto las imágenes no se guardan. Si se activa la opción, se guardan solo metadatos (timestamp, predicciones) en un CSV local.
    
    **¿Cómo conectar a Google Sheets?**  
    Subí el JSON de servicio a la raíz del proyecto y pon la ruta en la variable `GSHEET_CREDENTIALS_PATH` en app.py.
    Asegurate de que la cuenta de servicio tenga permisos para crear/editar la hoja.
    
    **¿Por qué los resultados pueden ser erráticos?**  
    Microexpresiones son sutiles y dependen de la calidad de la foto, iluminación, pose y diversidad cultural. Este sistema es exploratorio.
    """)

st.markdown("---")

uploaded_file = st.file_uploader("Subí una imagen (png/jpg/jpeg)", type=['png','jpg','jpeg'])
col1, col2 = st.columns([1,2])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error("Error al abrir la imagen. Asegurate de subir un archivo válido.")
        image = None
else:
    image = None

if image is not None:
    with col2:
        st.image(image, caption="Imagen cargada", use_column_width=True)

    # Opciones de procesamiento
    col_opt1, col_opt2, col_opt3 = st.columns([1,1,1])
    with col_opt1:
        save_local_checkbox = st.checkbox("Guardar resultado localmente (CSV)", value=True)
    with col_opt2:
        enable_gs_checkbox = st.checkbox("Subir resultados a Google Sheets (configurar credenciales)", value=False)
    with col_opt3:
        show_fac = st.checkbox("Mostrar explicación FAC (AUs)", value=True)

    if st.button("🔍 Analizar imagen"):
        if model is None:
            st.error("El modelo no está cargado. Revisa el sidebar y la variable MODEL_PATH.")
        else:
            with st.spinner("Analizando..."):
                emotions = predict_from_image(image, model, device)
                sd3 = compute_sd3_from_emotions(emotions)
                fac_text, emo_name, emo_prob, aus = explain_fac(emotions)

                # Mostrar resultados
                st.success("✅ Análisis completado")
                # Panel de resultado principal
                st.markdown(f"""
                <div class="conclusion-box">
                    <h2>🔬 Resultado principal</h2>
                    <p class="emotion-dominant">Microexpresión predominante: <strong>{emo_name}</strong> ({emo_prob*100:.1f}%)</p>
                    <p><strong>Rasgo SD3 predominante (heurístico):</strong> {max(sd3, key=sd3.get)} ({sd3[max(sd3, key=sd3.get)]:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

                # Explicación FAC
                if show_fac:
                    st.markdown("### 🔎 Explicación FAC (interpretación basada en AUs)")
                    st.markdown(fac_text)

                # Gráficos y tablas
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📊 Probabilidades (microexpresiones)")
                    df_em = pd.DataFrame({
                        "Emoción": list(emotions.keys()),
                        "Probabilidad": list(emotions.values())
                    })
                    df_plot = df_em.set_index("Emoción")
                    st.bar_chart(df_plot)
                    st.write(df_em.sort_values("Probabilidad", ascending=False).assign(Probabilidad=lambda d: d["Probabilidad"]*100).rename(columns={"Probabilidad":"%"}))
                with c2:
                    st.subheader("🧾 SD3 (heurístico)")
                    df_sd3 = pd.DataFrame({
                        "Rasgo": list(sd3.keys()),
                        "Puntaje": list(sd3.values())
                    }).set_index("Rasgo")
                    st.bar_chart(df_sd3)
                    st.write(df_sd3)

                # Interpretación textual breve
                st.markdown("### 🧩 Interpretación breve (automática)")
                # Reusar la función de análisis cruzado (puedes expandir con más reglas)
                # Aquí devolvemos un texto simple derivado de la emoción + SD3 dominante
                emoc_dom = emo_name
                ras_dom = max(sd3, key=sd3.get)
                interpretation = f"La microexpresión mayoritaria es **{emoc_dom}** y el rasgo SD3 con mayor puntaje es **{ras_dom}**. Esto sugiere una posible relación entre la expresión involuntaria detectada y tendencias de personalidad (esta interpretación es exploratoria y no clínica)."
                st.info(interpretation)

                # Guardar resultado (registro)
                timestamp = datetime.datetime.utcnow().isoformat()
                record = {
                    "timestamp_utc": timestamp,
                    "file_name": getattr(uploaded_file, "name", "uploaded_image"),
                    **{f"prob_{k}": float(v) for k, v in emotions.items()},
                    **{f"sd3_{k}": float(v) for k, v in sd3.items()},
                    "dominant_emotion": emo_name,
                    "dominant_emotion_prob": float(emo_prob)
                }

                if save_local_checkbox:
                    try:
                        save_result_local(record, LOCAL_RESULTS_CSV)
                        st.success(f"Resultado guardado localmente en `{LOCAL_RESULTS_CSV}`")
                    except Exception as e:
                        st.error(f"No se pudo guardar localmente: {e}")

                if enable_gs_checkbox:
                    if GSHEET_CREDENTIALS_PATH and GS_AVAILABLE:
                        try:
                            upload_to_gsheets(record, GSHEET_CREDENTIALS_PATH, GSHEET_NAME)
                            st.success("Resultado subido a Google Sheets correctamente.")
                        except Exception as e:
                            st.error(f"Error subiendo a Google Sheets: {e}")
                    else:
                        st.error("Subida a Google Sheets no activada. Revisa GSHEET_CREDENTIALS_PATH y las librerías instaladas.")

                # Advertencia ética
                st.markdown("---")
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Nota importante:</strong> Esta aplicación proporciona una evaluación exploratoria basada en datos visuales y heurísticos derivados del SD3. 
                    No debe usarse para diagnóstico ni para tomar decisiones que afecten a las personas. Los resultados son probabilísticos y dependen de la calidad de la imagen, el contexto cultural y las limitaciones del modelo.
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("👆 Sube una imagen para comenzar el análisis")
    st.markdown("---")
    st.markdown("### 🎯 ¿Qué hace esta aplicación?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 😊 Detección de Microexpresiones")
        st.write("Identifica 7 clases: Alegría, Tristeza, Enojo, Sorpresa, Miedo, Disgusto y Neutral.")
    with col2:
        st.markdown("#### 🧠 SD3 (heurístico)")
        st.write("Convierte la distribución de emociones en puntajes exploratorios de Maquiavelismo, Narcisismo y Psicopatía.")
    with col3:
        st.markdown("#### 🔬 Explicación FAC")
        st.write("Proporciona una interpretación basada en unidades de acción facial (AUs) típicas por emoción.")

# --------------------------
# FOOTER: enlaces / ayuda rápida
# --------------------------
st.markdown("---")
st.markdown("**Documentación rápida:** Si quieres integrar a Google Sheets sube el JSON de servicio y fija `GSHEET_CREDENTIALS_PATH` en app.py. Para convertir archivos Keras a PyTorch, conviene exportar a ONNX o re-entrenar en PyTorch y guardar `state_dict()` como .pth.")
st.markdown("**Responsabilidad:** DarkLens es una herramienta experimental de investigación. No reemplaza evaluación clínica ni diagnóstico profesional.")
