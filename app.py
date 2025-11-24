import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import pandas as pd
import os

# -----------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------
st.set_page_config(
    page_title="DarkLens",
    page_icon="🟣",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at center, #3a0066, #14001f);
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: white !important;
    }
    .stButton>button {
        background: #6a0dad !important;
        color: white !important;
        border-radius: 8px !important;
        width: 100%;
        padding: 0.5rem;
    }
    .conclusion-box {
        background: rgba(168, 85, 247, 0.2);
        border-left: 4px solid #a855f7;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .emotion-dominant {
        font-size: 1.5rem;
        font-weight: bold;
        color: #a855f7;
    }
    .warning-box {
        background: rgba(236, 72, 153, 0.2);
        border-left: 4px solid #ec4899;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------
# MODELO
# -----------------------------------------------------
class MicroExpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def cargar_modelo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MicroExpNet()

    try:
        state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
        first_key = list(state.keys())[0]

        if first_key.startswith("model.model."):
            new_state = {k.replace("model.", "", 1): v for k, v in state.items()}
            model.load_state_dict(new_state, strict=True)

        elif first_key.startswith("model."):
            new_state = {k.replace("model.", ""): v for k, v in state.items()}
            model.model.load_state_dict(new_state, strict=True)

        else:
            model.model.load_state_dict(state, strict=True)

    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {e}")
        raise

    model.to(device)
    model.eval()
    return model, device


model, device = cargar_modelo()

# -----------------------------------------------------
# PREPROCESAMIENTO
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels = ["Alegría", "Tristeza", "Enojo", "Sorpresa", "Miedo", "Disgusto", "Neutral"]


# -----------------------------------------------------
# FAC (análisis muscular explicativo)
# -----------------------------------------------------
FAC_MAP = {
    "Alegría": ["AU6 (mejillas elevadas)", "AU12 (comisuras elevadas)"],
    "Tristeza": ["AU1 (cejas internas elevadas)", "AU15 (comisuras caídas)"],
    "Enojo": ["AU4 (ceño fruncido)", "AU7 (párpados tensos)"],
    "Sorpresa": ["AU1+2 (cejas elevadas)", "AU5 (ojos muy abiertos)", "AU26 (boca abierta)"],
    "Miedo": ["AU1+2 (cejas elevadas)", "AU5 (ojos muy abiertos)", "AU20 (labios tensos)"],
    "Disgusto": ["AU9 (arruga de nariz)", "AU10 (labio superior elevado)"],
    "Neutral": ["Sin activaciones destacadas"]
}


# -----------------------------------------------------
# SD3 SIMULADO EN FUNCIÓN DE EMOCIONES
# -----------------------------------------------------
def compute_sd3(em):
    maqu = em["Enojo"] * 0.6 + em["Disgusto"] * 0.4
    narc = em["Alegría"] * 0.5 + em["Neutral"] * 0.5
    psic = em["Miedo"] * 0.7 + em["Sorpresa"] * 0.3

    return {
        "Maquiavelismo": round(maqu * 100, 2),
        "Narcisismo": round(narc * 100, 2),
        "Psicopatía": round(psic * 100, 2)
    }


# -----------------------------------------------------
# ANÁLISIS
# -----------------------------------------------------
def analizar(image):
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    emotions = {labels[i]: float(probs[i]) for i in range(7)}
    sd3 = compute_sd3(emotions)

    return emotions, sd3


# -----------------------------------------------------
# INTERFAZ
# -----------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🟣 DarkLens — Detector de Microexpresiones</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Subí una imagen", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("🔍 Analizar imagen"):
        emotions, sd3 = analizar(image)

        # Emoción dominante
        emo_dom = max(emotions, key=emotions.get)

        st.success("✅ Análisis completado!")

        # Bloque principal
        st.markdown(f"""
        <div class="conclusion-box">
            <h2>🔬 Análisis Psicológico Completo</h2>
            <p class="emotion-dominant">
                Emoción predominante: {emo_dom}
            </p>
            <h3>📌 Activaciones FACS estimadas:</h3>
            <ul>
                {''.join([f"<li>{au}</li>" for au in FAC_MAP[emo_dom]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Mostrar emociones
        st.subheader("📊 Microexpresiones detectadas")
        df_em = pd.DataFrame({"Emoción": emotions.keys(), "Probabilidad": emotions.values()})
        st.bar_chart(df_em.set_index("Emoción"))

        # Mostrar SD3
        st.subheader("🧠 Estimación SD3")
        df_sd = pd.DataFrame({"Rasgo": sd3.keys(), "Puntaje": sd3.values()})
        st.bar_chart(df_sd.set_index("Rasgo"))

        st.markdown("""
        <div class="warning-box">
            ⚠️ Esta aplicación NO realiza diagnósticos clínicos.<br>
            El análisis es experimental, académico y orientativo.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👆 Subí una imagen para comenzar el análisis.")
