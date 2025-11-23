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
@st.cache_resource
def cargar_modelo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MicroExpNet()
    
    try:
        # Intentar cargar directamente
        state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
        
        # Verificar estructura del state_dict
        if 'model.model.features.0.0.weight' in state:
            # Si tiene prefijo "model.", quitarlo
            new_state = {k.replace("model.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state, strict=True)
        elif 'model.features.0.0.weight' in state:
            # Si tiene estructura normal
            model.model.load_state_dict(state, strict=True)
        else:
            # Intentar cargar directamente
            model.load_state_dict(state, strict=True)
            
        st.success(f"✅ Modelo cargado correctamente en {device}")
        
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {str(e)}")
        st.info("Verifica que el archivo microexp_retrained_FER2013.pth esté en la raíz del proyecto")
        raise
    
    model.to(device)
    model.eval()
    
    return model, device
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
def analisis_personalidad_completo(emotions, sd3):
    """Genera un análisis psicológico completo"""
    
    # Determinar emoción dominante
    emocion_dominante = max(emotions.items(), key=lambda x: x[1])
    emo_nombre, emo_valor = emocion_dominante
    
    # Determinar rasgo SD3 dominante
    rasgo_dominante = max(sd3.items(), key=lambda x: x[1])
    rasgo_nombre, rasgo_valor = rasgo_dominante
    
    # Análisis cruzado: Emoción + Rasgo
    analisis_cruzado = {
        # Maquiavelismo
        ("Enojo", "Maquiavelismo"): "Combinación de enojo y maquiavelismo sugiere una persona estratégica que puede usar la confrontación como herramienta para sus objetivos. Tiende a ser directa cuando le conviene.",
        ("Neutral", "Maquiavelismo"): "La neutralidad emocional combinada con maquiavelismo indica control calculado. Esta persona oculta sus verdaderas intenciones y mantiene una fachada serena para manipular situaciones.",
        ("Alegría", "Maquiavelismo"): "La alegría con maquiavelismo puede indicar carisma estratégico. Usa el encanto para influir en otros y lograr sus metas, mostrándose amigable cuando es útil.",
        ("Disgusto", "Maquiavelismo"): "El disgusto con maquiavelismo sugiere rechazo calculado. Esta persona puede descartar relaciones o situaciones que no le benefician sin remordimientos.",
        
        # Narcisismo
        ("Alegría", "Narcisismo"): "Alegría narcisista refleja autoestima elevada y búsqueda de admiración. La felicidad está ligada al reconocimiento externo y la validación constante.",
        ("Sorpresa", "Narcisismo"): "Sorpresa narcisista puede indicar reacciones dramáticas. Esta persona amplifica sus respuestas emocionales para llamar la atención y ser el centro de las situaciones.",
        ("Enojo", "Narcisismo"): "Enojo narcisista sugiere 'ira narcisista' - reacciones intensas ante críticas o falta de reconocimiento. El ego herido genera respuestas desproporcionadas.",
        ("Neutral", "Narcisismo"): "Neutralidad narcisista puede reflejar frialdad calculada. Mantiene distancia emocional para proyectar superioridad y control sobre otros.",
        
        # Psicopatía
        ("Neutral", "Psicopatía"): "Neutralidad psicopática indica aplanamiento afectivo. Baja reactividad emocional, procesamiento frío de situaciones, y dificultad para conectar emocionalmente.",
        ("Alegría", "Psicopatía"): "Alegría psicopática puede ser superficial y calculada. Las expresiones positivas no reflejan conexión emocional genuina, sino respuestas aprendidas socialmente.",
        ("Miedo", "Psicopatía"): "Miedo con psicopatía es inusual - puede indicar ansiedad situacional sin procesamiento emocional profundo. La respuesta es más cognitiva que afectiva.",
        ("Sorpresa", "Psicopatía"): "Sorpresa psicopática refleja reactividad baja. Incluso situaciones inesperadas generan respuestas emocionales atenuadas, manteniendo el control.",
    }
    
    # Buscar combinación específica
    clave = (emo_nombre, rasgo_nombre)
    analisis_especifico = analisis_cruzado.get(
        clave,
        f"La combinación de {emo_nombre.lower()} y {rasgo_nombre.lower()} sugiere un perfil emocional complejo que requiere análisis más profundo."
    )
    
    # Nivel de intensidad
    if rasgo_valor > 60:
        nivel = "marcado"
    elif rasgo_valor > 40:
        nivel = "moderado"
    else:
        nivel = "leve"
    
    return {
        "emocion_dominante": emo_nombre,
        "emocion_valor": emo_valor * 100,
        "rasgo_dominante": rasgo_nombre,
        "rasgo_valor": rasgo_valor,
        "nivel_intensidad": nivel,
        "analisis_especifico": analisis_especifico
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
