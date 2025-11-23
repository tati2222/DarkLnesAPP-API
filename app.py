import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import pandas as pd
import os

# --------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------
st.set_page_config(
    page_title="DarkLens",
    page_icon="🟣",
    layout="wide"
)

# CSS personalizado
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
</style>
""", unsafe_allow_html=True)

# --------------------------
# CARGA DEL MODELO
# --------------------------
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
    
    # Cargar pesos entrenados
    state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
    new_state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    
    return model, device

model, device = cargar_modelo()

# --------------------------
# PREPROCESAMIENTO
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

labels = ["Alegría", "Tristeza", "Enojo", "Sorpresa", "Miedo", "Disgusto", "Neutral"]

# --------------------------
# SD3
# --------------------------
def compute_sd3(emotions):
    maqu = emotions["Enojo"] * 0.6 + emotions["Disgusto"] * 0.4
    narc = emotions["Alegría"] * 0.5 + emotions["Neutral"] * 0.5
    psic = emotions["Miedo"] * 0.7 + emotions["Sorpresa"] * 0.3
    
    return {
        "Maquiavelismo": round(maqu * 100, 2),
        "Narcisismo": round(narc * 100, 2),
        "Psicopatía": round(psic * 100, 2)
    }

# --------------------------
# ANÁLISIS
# --------------------------
def analizar(image):
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    
    emotions = {labels[i]: float(probs[i]) for i in range(7)}
    sd3 = compute_sd3(emotions)
    
    return emotions, sd3

# --------------------------
# INTERFAZ
# --------------------------
st.markdown("<h1 style='text-align:center;'>🟣 DarkLens — Detector de Microexpresiones</h1>", unsafe_allow_html=True)

# Subir imagen
uploaded_file = st.file_uploader("Subí una imagen", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Imagen cargada', use_column_width=True)
    
    # Botón de análisis
    if st.button('🔍 Analizar imagen'):
        with st.spinner('Analizando microexpresiones...'):
            emotions, sd3 = analizar(image)
        
        st.success('✅ Análisis completado!')
        
        # Crear dos columnas para los gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Microexpresiones detectadas")
            # Convertir a DataFrame para graficar
            df_emotions = pd.DataFrame({
                'Emoción': list(emotions.keys()),
                'Probabilidad': list(emotions.values())
            })
            st.bar_chart(df_emotions.set_index('Emoción'))
            
            # Mostrar valores exactos
            st.write("**Valores:**")
            for emo, val in emotions.items():
                st.write(f"- {emo}: {val*100:.2f}%")
        
        with col2:
            st.subheader("Rasgos SD3")
            # Convertir a DataFrame para graficar
            df_sd3 = pd.DataFrame({
                'Rasgo': list(sd3.keys()),
                'Puntuación': list(sd3.values())
            })
            st.bar_chart(df_sd3.set_index('Rasgo'))
            
            # Mostrar valores exactos
            st.write("**Valores:**")
            for rasgo, val in sd3.items():
                st.write(f"- {rasgo}: {val}%")

else:
    st.info("👆 Sube una imagen para comenzar el análisis")
