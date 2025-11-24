# app.py - DarkLens (versión completa con FAQ, Ética y export opcional a Google Sheets)
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
import time
import json

# Optional Google Sheets integration (requires service account JSON and gspread)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GS_AVAILABLE = True
except Exception:
    GS_AVAILABLE = False

# --------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------
st.set_page_config(
    page_title="DarkLens",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (mantener la estética)
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at center, #3a0066, #14001f);
        }
        .stButton>button {
            background: #6a0dad !important;
            color: white !important;
            border-radius: 8px !important;
        }
        .conclusion-box {
            background: rgba(168, 85, 247, 0.12);
            border-left: 4px solid #a855f7;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: white;
        }
        .emotion-dominant {
            font-size: 1.35rem;
            font-weight: bold;
            color: #a855f7;
        }
        .warning-box {
            background: rgba(236, 72, 153, 0.12);
            border-left: 4px solid #ec4899;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: white;
        }
        .metric-box {
            background: rgba(255, 255, 255, 0.03);
            padding: 0.9rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            color: white;
        }
        .faq-box {
            background: rgba(255,255,255,0.02);
            padding: 0.8rem;
            border-radius: 6px;
            color: white;
        }
        .small-muted {
            color: rgba(255,255,255,0.65);
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# CONSTANTES Y RUTAS
# --------------------------
MODEL_FILENAME = "microexp_retrained_FER2013.pth"  # asegúrate que este archivo esté en la raíz del proyecto
SERVICE_ACCOUNT_JSON = "service_account.json"     # opcional: subir si querés exportar a Google Sheets
GOOGLE_SHEET_NAME = "DarkLens_Results"            # nombre de la hoja que se usará (si existe la credencial)

# Etiquetas del modelo (las mismas que usaste en entrenamiento)
LABELS = ["Alegría", "Tristeza", "Enojo", "Sorpresa", "Miedo", "Disgusto", "Neutral"]

# --------------------------
# HELPER: Estructura del modelo
# --------------------------
class MicroExpNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.model(x)

# --------------------------
# CARGA ROBUSTA DEL MODELO
# --------------------------
@st.cache_resource(ttl=3600)
def cargar_modelo_ruta(model_path: str):
    """Carga el modelo con tolerancia a prefijos en state_dict (model., model.model., directo)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicroExpNet(num_classes=len(LABELS))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo del modelo no se encontró en la ruta: {model_path}")
    # Cargar archivo
    state = torch.load(model_path, map_location=device)
    # Si viene dict tipo {'model_state_dict': {...}} o {'state_dict': {...}} tratamos varios casos
    # Normalizamos a un dict simple de pesos
    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
        key = 'model_state_dict' if 'model_state_dict' in state else 'state_dict'
        state = state[key]
    # state ahora debería ser un dict con claves
    if not isinstance(state, dict):
        raise RuntimeError("El contenido del archivo del modelo no es un state_dict reconocible.")
    # Ajustes de prefijos
    keys = list(state.keys())
    if not keys:
        raise RuntimeError("El state_dict del modelo está vacío.")
    first_key = keys[0]
    # Heurísticas para cargar
    try:
        if first_key.startswith("model.model."):
            # remover un 'model.' inicial
            new_state = {k.replace("model.", "", 1): v for k, v in state.items()}
            model.load_state_dict(new_state, strict=True)
        elif first_key.startswith("model."):
            # remover 'model.' y cargar directamente en submódulo model.model
            new_state = {k.replace("model.", "", 1): v for k, v in state.items()}
            # Si las claves ahora empiezan por 'model.' de nuevo, quitamos sólo una vez
            model.load_state_dict(new_state, strict=True)
        else:
            # intentamos cargar directo (compatibilidad normal)
            model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model, device
    except RuntimeError as e:
        # Intento más tolerante: cargar en modo flexible (no strict)
        try:
            # Probar cargar en model.model si existe
            temp_state = state
            # quitar prefijos comunes
            stripped = {}
            for k, v in temp_state.items():
                newk = k
                if k.startswith("module."):
                    newk = k.replace("module.", "", 1)
                if newk.startswith("model."):
                    newk = newk.replace("model.", "", 1)
                stripped[newk] = v
            model.load_state_dict(stripped, strict=False)
            model.to(device)
            model.eval()
            return model, device
        except Exception as e2:
            raise RuntimeError(f"Error cargando state_dict: {e} | intento alternativo falló: {e2}")

# Intentar cargar el modelo al iniciar la app
model_load_success = False
try:
    model, device = cargar_modelo_ruta(MODEL_FILENAME)
    model_load_success = True
except Exception as e:
    st.error(f"Error cargando modelo: {e}")
    st.info("Subí el archivo del modelo 'microexp_retrained_FER2013.pth' a la raíz del proyecto y recargá.")
    # Para que la app no rompa el import, definimos placeholders
    model = None
    device = torch.device("cpu")

# --------------------------
# TRANSFORM Y PREPROCESAMIENTO
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# MAPEOS FAC (simplificado, para interpretación)
# --------------------------
# Esto NO es un FACS completo. Es un mapeo orientativo para ayudar la explicación en la app.
FAC_MAPPING = {
    "Alegría": {
        "Aus": ["AU6 (Mejora de mejillas)", "AU12 (Elevación comisura labial)"],
        "Regiones": ["Comisura de la boca", "Mejillas"],
        "Descripción": "Sonrisa genuina: elevación de comisura y arrugas alrededor de los ojos cuando es auténtica."
    },
    "Tristeza": {
        "Aus": ["AU1 (Elevación de cejas internas)", "AU15 (Depresión comisura labial)"],
        "Regiones": ["Ceño", "Comisura de la boca"],
        "Descripción": "Párpados caídos y comisura bajada; mirada hacia abajo y tensión en párpados."
    },
    "Enojo": {
        "Aus": ["AU4 (Ceño fruncido)", "AU23 (Tensión labial)"],
        "Regiones": ["Entrecejo", "Mandíbula"],
        "Descripción": "Ceño fruncido y mandíbula tensa: indicadores de hostilidad o irritación."
    },
    "Sorpresa": {
        "Aus": ["AU1+AU2 (Cejas elevadas)", "AU5 (Apertura de ojos)"],
        "Regiones": ["Ceja", "Ojos"],
        "Descripción": "Cejas levantadas y ojos abiertos; boca puede abrirse levemente."
    },
    "Miedo": {
        "Aus": ["AU1+AU2 (Ceja elevada)", "AU20 (Tensión labios)"],
        "Regiones": ["Ojo", "Boca"],
        "Descripción": "Apertura ocular con tensión; la expresión puede parecer mezcla entre sorpresa y ansiedad."
    },
    "Disgusto": {
        "Aus": ["AU9 (Arrugamiento nariz)", "AU10 (Elevación labio superior)"],
        "Regiones": ["Nariz", "Labio superior"],
        "Descripción": "Arrugas en la nariz y levantamiento del labio superior, como rechazo."
    },
    "Neutral": {
        "Aus": ["Ausencia de AUs fuertes"],
        "Regiones": ["Rostro relajado"],
        "Descripción": "Rostro sin activación muscular significativa; puede indicar control o ausencia de emoción manifiesta."
    }
}

# --------------------------
# FUNCIONES DE ANÁLISIS PSICOLÓGICO (SD3 heurístico)
# --------------------------
def compute_sd3_from_emotions(emotions: dict):
    """
    Cálculo heurístico para generar 'puntuaciones SD3' a partir de probabilidades de emociones.
    Esto es un puente heurístico (no una validación clínica). Las fórmulas son simples combinaciones ponderadas.
    """
    maqu = emotions.get("Enojo", 0) * 0.6 + emotions.get("Disgusto", 0) * 0.4
    narc = emotions.get("Alegría", 0) * 0.5 + emotions.get("Neutral", 0) * 0.5
    psic = emotions.get("Miedo", 0) * 0.7 + emotions.get("Sorpresa", 0) * 0.3
    # Convertir a 0-100
    return {
        "Maquiavelismo": round(maqu * 100, 2),
        "Narcisismo": round(narc * 100, 2),
        "Psicopatía": round(psic * 100, 2)
    }

def analyze_cross(emotions: dict, sd3: dict):
    """
    Analiza y devuelve un texto interpretativo.
    Mantener este análisis como heurístico: explicaciones no clínicas.
    """
    dominante_emo = max(emotions.items(), key=lambda x: x[1])
    dominante_sd3 = max(sd3.items(), key=lambda x: x[1])

    emo_name, emo_val = dominante_emo
    sd3_name, sd3_val = dominante_sd3

    # Nivel
    if sd3_val > 65:
        nivel = "MARCADO"
        simbolo = "🔴"
    elif sd3_val > 40:
        nivel = "MODERADO"
        simbolo = "🟡"
    else:
        nivel = "LEVE"
        simbolo = "🟢"

    # Interpretación principal (texto resumido)
    interpretation = ""
    # Usamos una base simple de frases; esto puede editarse fácilmente
    if sd3_name == "Maquiavelismo":
        if emo_name == "Enojo":
            interpretation = ("La combinación de enojo con puntuación alta en maquiavelismo sugiere "
                              "una predisposición a utilizar la confrontación como herramienta estratégica. "
                              "Se debe interpretar con precaución y en contexto.")
        elif emo_name == "Neutral":
            interpretation = ("Neutralidad facial con alto maquiavelismo indica control emocional calculado: "
                              "la persona puede ocultar intenciones reales detrás de una fachada serena.")
        else:
            interpretation = ("Combinación de emociones con maquiavelismo que sugiere comportamiento estratégico; "
                              "interpretar en contexto.")
    elif sd3_name == "Narcisismo":
        interpretation = ("Patrón compatible con búsqueda de validación externa. Si la emoción dominante es positiva, "
                          "puede corresponder a expresividad orientada a recibir atención y aprobación.")
    elif sd3_name == "Psicopatía":
        interpretation = ("Patrón que puede asociarse a reactividad emocional atenuada. Interpretar con cautela: "
                          "no implica juicio clínico; la expresión puede ser instrumental o superficial.")
    else:
        interpretation = ("Perfil complejo: requiere análisis complementario con SD3 y datos conductuales.")
    return {
        "emocion_dominante": (emo_name, emo_val),
        "rasgo_dominante": (sd3_name, sd3_val),
        "nivel": nivel,
        "simbolo": simbolo,
        "texto": interpretation
    }

# --------------------------
# FUNCIONES PARA EXPORTAR (GOOGLE SHEETS) - OPCIONAL
# --------------------------
def export_to_google_sheets(row_dict: dict, cred_path=SERVICE_ACCOUNT_JSON, sheet_name=GOOGLE_SHEET_NAME):
    """
    Exporta un diccionario como fila a Google Sheets. 
    Requiere subir a la raíz un service_account.json con permisos y compartir la sheet con el client_email.
    """
    if not GS_AVAILABLE:
        raise RuntimeError("gspread o google oauth no están instalados en el entorno. Instalá gspread y google-auth.")
    if not os.path.exists(cred_path):
        raise FileNotFoundError("No se encontró el archivo de credenciales service_account.json en la raíz.")
    # Autenticación
    creds = Credentials.from_service_account_file(cred_path, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    gc = gspread.authorize(creds)
    # Abrir / crear hoja
    try:
        sh = gc.open(sheet_name)
    except Exception:
        sh = gc.create(sheet_name)
        # Nota: podés necesitar compartir manualmente la sheet o configurar permisos
    worksheet = None
    try:
        worksheet = sh.sheet1
    except Exception:
        worksheet = sh.add_worksheet(title="Sheet1", rows="1000", cols="20")
    # Escribir encabezados si está vacío
    values = list(row_dict.values())
    keys = list(row_dict.keys())
    if worksheet.row_count == 0 or worksheet.get_all_values() == []:
        worksheet.append_row(keys)
    worksheet.append_row(values)
    return True

# --------------------------
# PREDICCIÓN PRINCIPAL
# --------------------------
def predict_emotions_from_image(pil_image: Image.Image):
    if model is None:
        raise RuntimeError("El modelo no está cargado. Subí el archivo .pth y recargá la app.")
    img = pil_image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    # Crear diccionario de probabilidades
    emotions = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return emotions

# --------------------------
# BARRA LATERAL: FAQ, ETICA, CONFIGS
# --------------------------
st.sidebar.title("DarkLens — Panel")

with st.sidebar.expander("⚖️ Ética y privacidad (resumen)"):
    st.markdown(
        """
        **Principios claves aplicados en DarkLens**:
        - Uso de datasets públicos para entrenamiento.  
        - Consentimiento informado requerido para datos voluntarios.  
        - No se utiliza para diagnóstico clínico.  
        - Minimización de datos: opcionalmente no guardamos imágenes.  
        - Transparencia: se informa la heurística SD3 y las limitaciones.
        """)
    if st.button("Ver apartado de ética completo"):
        st.markdown("""
        ### Ética completa — puntos destacados
        1. **No diagnóstico**: DarkLens no es una herramienta clínica ni forense. Sus salidas son probabilísticas y exploratorias.  
        2. **Consentimiento**: cualquier recolección de imágenes de voluntarios debe incluir un consentimiento informado que explique fines, duración del almacenamiento y derechos de acceso/retirada.  
        3. **Datos sensibles**: las imágenes faciales se consideran biométricas; se debe proteger su acceso mediante cifrado en caso de almacenamiento.  
        4. **Sesgos**: modelos entrenados en corpora no representativos pueden reproducir sesgos culturales y demográficos. Se recomienda reportar limitaciones de cobertura demográfica.  
        5. **Transparencia**: publicar procedimientos, arquitecturas y métricas (accuracy, Balanced Accuracy, Kappa, MCC) para reproducibilidad.  
        6. **Prohibición de usos**: no usar para toma de decisiones legales, laborales o médicas.  
        (Esta es una síntesis; en el informe se desarrolla cada punto con referencias.)
        """)
with st.sidebar.expander("❓ FAQ / Preguntas frecuentes (útil para la app)"):
    st.markdown("<div class='faq-box'>", unsafe_allow_html=True)
    st.markdown("**¿Qué hace DarkLens?**")
    st.markdown("Detecta microexpresiones en una imagen (7 clases) y genera una interpretación heurística combinada con una estimación SD3 (no clínica).")
    st.markdown("**¿Es un diagnóstico?**")
    st.markdown("No. Es una herramienta experimental y exploratoria. No sustituye evaluación clínica profesional.")
    st.markdown("**¿Puedo subir cualquier foto?**")
    st.markdown("Se recomiendan fotos frontales, con buena iluminación y sin occlusiones. No subir imágenes de terceros sin consentimiento.")
    st.markdown("**¿Dónde guardan mis datos?**")
    st.markdown("Por defecto esta demo no guarda las imágenes en servidores. Si activás la exportación a Google Sheets, se guardarán resultados numéricos (no imágenes).")
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar.expander("⚙️ Configuraciones (export)"):
    use_gs = st.checkbox("Habilitar export a Google Sheets (requiere service_account.json)", value=False)
    if use_gs:
        if not GS_AVAILABLE:
            st.warning("No está instalada la librería gspread/google-auth en este entorno. Instálala en requirements.txt: gspread, google-auth")
        else:
            st.info("Asegurate de subir 'service_account.json' en la carpeta raíz y de compartir la hoja con el service account.")
    show_facs = st.checkbox("Mostrar explicación FAC (regiones/AUs) en informe", value=True)

# --------------------------
# INTERFAZ PRINCIPAL
# --------------------------
st.markdown("<h1 style='text-align:center; color:white;'>🟣 DarkLens — Detector de Microexpresiones</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:rgba(255,255,255,0.8)'>Subí una imagen frontal y obtén una predicción de microexpresión + interpretación heurística con SD3</p>", unsafe_allow_html=True)
st.markdown("---")

col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    uploaded_file = st.file_uploader("Subí una imagen (jpg, png)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

    if uploaded_file is not None:
        # Conversión a PIL
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen cargada", use_column_width=True)
        st.markdown("**Previsualización lista — presiona Analizar**")

        if st.button("🔍 Analizar imagen"):
            # Analizar
            start_time = time.time()
            try:
                emotions = predict_emotions_from_image(image)
            except Exception as e:
                st.error(f"Error en predicción: {e}")
                emotions = None

            if emotions:
                sd3 = compute_sd3_from_emotions(emotions)
                cross = analyze_cross(emotions, sd3)

                # Mostrar resumen
                st.success("✅ Análisis completado")
                with st.container():
                    st.markdown(
                        f"""
                        <div class="conclusion-box">
                        <h2>🔬 Resultado — Análisis integrado</h2>
                        <p class="emotion-dominant">Emoción dominante: <strong>{cross['emocion_dominante'][0]}</strong> ({cross['emocion_dominante'][1]*100:.1f}%) &nbsp;&nbsp;|&nbsp;&nbsp;
                        Rasgo SD3 dominante: <strong>{cross['rasgo_dominante'][0]}</strong> ({cross['rasgo_dominante'][1]:.1f}%)</p>
                        <p><strong>Nivel del rasgo:</strong> {cross['simbolo']} {cross['nivel']}</p>
                        <hr style="border-color: rgba(255,255,255,0.2); margin: 0.6rem 0;">
                        <p style="line-height:1.6;">{cross['texto']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Mostrar gráfico de barras con probabilidades
                df_em = pd.DataFrame({
                    "Emoción": list(emotions.keys()),
                    "Prob": [v*100 for v in emotions.values()]
                }).sort_values("Prob", ascending=False)
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.subheader("📊 Probabilidades (microexpresiones)")
                    st.bar_chart(df_em.set_index("Emoción"))
                    st.write(df_em.to_html(index=False), unsafe_allow_html=True)
                with col_b:
                    st.subheader("🧾 SD3 (heurístico)")
                    df_sd = pd.DataFrame({"Rasgo": list(sd3.keys()), "Valor": list(sd3.values())}).sort_values("Valor", ascending=False)
                    st.bar_chart(df_sd.set_index("Rasgo"))
                    st.write(df_sd.to_html(index=False), unsafe_allow_html=True)

                # Explicación FAC si está activado
                if show_facs:
                    emo_dom = cross['emocion_dominante'][0]
                    mapping = FAC_MAPPING.get(emo_dom, {})
                    st.markdown("---")
                    st.markdown(f"### 🎯 Explicación facial (FAC orientativo) — {emo_dom}")
                    st.markdown(f"**Regiones implicadas:** {', '.join(mapping.get('Regiones', ['-']))}")
                    st.markdown(f"**Unidades de acción (AU) típicas:** {', '.join(mapping.get('Aus', ['-']))}")
                    st.markdown(f"**Descripción:** {mapping.get('Descripcion', mapping.get('Descripción', 'Explicación no disponible'))}")
                    st.markdown("---")

                # Interpretación detallada SD3
                st.markdown("### 🔍 Interpretación detallada (SD3)")
                def pretty_sd3_interpret(sd3dict):
                    rows = []
                    for k, v in sd3dict.items():
                        level = "Bajo"
                        if v > 65:
                            level = "Alto"
                        elif v > 40:
                            level = "Moderado"
                        rows.append((k, f"{v:.1f}", level))
                    return rows
                rows = pretty_sd3_interpret(sd3)
                for r in rows:
                    st.markdown(f"<div class='metric-box'><strong>{r[0]}:</strong> {r[1]}% — {r[2]}</div>", unsafe_allow_html=True)

                # Etiqueta de advertencia (ética)
                st.markdown("""
                <div class="warning-box">
                <strong>⚠️ Aviso:</strong> DarkLens es una herramienta de investigación. **No es diagnóstico clínico ni forense.**
                Interpreta los resultados con cautela y en su contexto clínico/psicológico adecuado.
                </div>
                """, unsafe_allow_html=True)

                # Botón para descargar JSON con resultados
                result_payload = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    "emotions": emotions,
                    "sd3": sd3,
                    "dominant_emotion": cross['emocion_dominante'],
                    "dominant_sd3": cross['rasgo_dominante'],
                    "interpretation": cross['texto']
                }
                buf = io.BytesIO()
                buf.write(json.dumps(result_payload, indent=2).encode("utf-8"))
                buf.seek(0)
                st.download_button("⬇️ Descargar resultado (JSON)", data=buf, file_name="darklens_result.json", mime="application/json")

                # Exportar a Google Sheets si está activo
                if use_gs:
                    if GS_AVAILABLE and os.path.exists(SERVICE_ACCOUNT_JSON):
                        try:
                            # Preparar fila
                            row = {
                                "timestamp": result_payload["timestamp"],
                                "dominant_emotion": cross['emocion_dominante'][0],
                                "dominant_emotion_prob": f"{cross['emocion_dominante'][1]:.4f}",
                                "dominant_sd3": cross['rasgo_dominante'][0],
                                "dominant_sd3_val": f"{cross['rasgo_dominante'][1]:.2f}",
                                "emotions_json": json.dumps(emotions),
                                "sd3_json": json.dumps(sd3)
                            }
                            ok = export_to_google_sheets(row, cred_path=SERVICE_ACCOUNT_JSON, sheet_name=GOOGLE_SHEET_NAME)
                            if ok:
                                st.success("✅ Resultados exportados a Google Sheets correctamente.")
                        except Exception as e:
                            st.error(f"Error exportando a Google Sheets: {e}")
                            st.info("Verifica gspread, service_account.json y permisos del service account.")
                    else:
                        st.warning("No se puede exportar: falta gspread o service_account.json en la raíz.")

                elapsed = time.time() - start_time
                st.caption(f"Procesado en {elapsed:.2f} s (sin GPU puede tardar más).")

            else:
                st.error("No se pudieron obtener probabilidades del modelo.")
    else:
        st.info("👆 Subí una imagen frontal para comenzar el análisis. Recomendado: buena iluminación y cara despejada.")
        st.markdown("---")
        st.markdown("###  Guía rápida")
        st.markdown("- Usa fotos frontales sin demasiado recorte.\n- Evita filtros, lentes oscuros o manos que tapen el rostro.\n- Esta demo no guarda la imagen por defecto.")
        st.markdown("---")

# --------------------------
# PIE / INFO ADICIONAL
# --------------------------
st.markdown("<hr style='border-color: rgba(255,255,255,0.08)'>", unsafe_allow_html=True)
with st.expander("📚 Fuentes y notas metodológicas (resumen)"):
    st.markdown("""
    Este proyecto combina teoría emocional (Ekman, Barrett, Matsumoto), medidas de personalidad (Short Dark Triad - Jones & Paulhus),
    y modelos de visión por computador (EfficientNet) para estudio exploratorio. Las interpretaciones son heurísticas y orientativas.
    Para mayor detalle, consultá la bibliografía del informe.
    """)

# --------------------------
# FIN
# --------------------------
