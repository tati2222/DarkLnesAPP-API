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
    
    try:
        # Cargar el state_dict
        state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
        
        # Verificar estructura del state_dict
        first_key = list(state.keys())[0]
        
        if first_key.startswith('model.model.'):
            # Si tiene doble prefijo "model.model.", quitarlo
            new_state = {k.replace("model.", "", 1): v for k, v in state.items()}
            model.load_state_dict(new_state, strict=True)
            st.success(f"✅ Modelo cargado (formato: model.model.*) en {device}")
        elif first_key.startswith('model.'):
            # Si tiene prefijo "model.", quitarlo
            new_state = {k.replace("model.", ""): v for k, v in state.items()}
            model.model.load_state_dict(new_state, strict=True)
            st.success(f"✅ Modelo cargado (formato: model.*) en {device}")
        else:
            # Sin prefijo, cargar directo al submodelo
            model.model.load_state_dict(state, strict=True)
            st.success(f"✅ Modelo cargado (formato directo) en {device}")
            
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {str(e)}")
        st.info("Verifica que el archivo microexp_retrained_FER2013.pth esté en la raíz del proyecto")
        if 'state' in locals():
            st.info(f"Primera clave del state_dict: {list(state.keys())[0]}")
        raise
    
    model.to(device)
    model.eval()
    
    return model, device

# Cargar modelo al inicio
model, device = cargar_modelo()

# --------------------------
# PREPROCESAMIENTO
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels = ["Alegría", "Tristeza", "Enojo", "Sorpresa", "Miedo", "Disgusto", "Neutral"]

# --------------------------
# SD3 Y FUNCIONES DE ANÁLISIS
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
        ("Tristeza", "Maquiavelismo"): "La tristeza con maquiavelismo puede ser una máscara estratégica para generar empatía o manipular situaciones a su favor.",
        ("Sorpresa", "Maquiavelismo"): "La sorpresa con maquiavelismo sugiere adaptabilidad calculada ante situaciones inesperadas.",
        ("Miedo", "Maquiavelismo"): "El miedo con maquiavelismo indica precaución estratégica y evaluación de riesgos antes de actuar.",
        
        # Narcisismo
        ("Alegría", "Narcisismo"): "Alegría narcisista refleja autoestima elevada y búsqueda de admiración. La felicidad está ligada al reconocimiento externo y la validación constante.",
        ("Sorpresa", "Narcisismo"): "Sorpresa narcisista puede indicar reacciones dramáticas. Esta persona amplifica sus respuestas emocionales para llamar la atención y ser el centro de las situaciones.",
        ("Enojo", "Narcisismo"): "Enojo narcisista sugiere 'ira narcisista' - reacciones intensas ante críticas o falta de reconocimiento. El ego herido genera respuestas desproporcionadas.",
        ("Neutral", "Narcisismo"): "Neutralidad narcisista puede reflejar frialdad calculada. Mantiene distancia emocional para proyectar superioridad y control sobre otros.",
        ("Tristeza", "Narcisismo"): "La tristeza narcisista puede indicar depresión relacionada con falta de validación externa o heridas al ego.",
        ("Disgusto", "Narcisismo"): "El disgusto narcisista sugiere desprecio hacia quienes no reconocen su superioridad percibida.",
        ("Miedo", "Narcisismo"): "El miedo narcisista puede reflejar temor al rechazo, crítica o pérdida de estatus social.",
        
        # Psicopatía
        ("Neutral", "Psicopatía"): "Neutralidad psicopática indica aplanamiento afectivo. Baja reactividad emocional, procesamiento frío de situaciones, y dificultad para conectar emocionalmente.",
        ("Alegría", "Psicopatía"): "Alegría psicopática puede ser superficial y calculada. Las expresiones positivas no reflejan conexión emocional genuina, sino respuestas aprendidas socialmente.",
        ("Miedo", "Psicopatía"): "Miedo con psicopatía es inusual - puede indicar ansiedad situacional sin procesamiento emocional profundo. La respuesta es más cognitiva que afectiva.",
        ("Sorpresa", "Psicopatía"): "Sorpresa psicopática refleja reactividad baja. Incluso situaciones inesperadas generan respuestas emocionales atenuadas, manteniendo el control.",
        ("Enojo", "Psicopatía"): "El enojo psicopático tiende a ser instrumental y controlado, usado como herramienta más que como emoción genuina.",
        ("Tristeza", "Psicopatía"): "La tristeza psicopática es rara y superficial, sin la profundidad emocional típica de esta emoción.",
        ("Disgusto", "Psicopatía"): "El disgusto psicopático puede manifestarse como indiferencia fría ante situaciones que otros encontrarían repulsivas.",
    }
    
    # Buscar combinación específica
    clave = (emo_nombre, rasgo_nombre)
    analisis_especifico = analisis_cruzado.get(
        clave,
        f"La combinación de {emo_nombre.lower()} con {rasgo_nombre.lower()} elevado sugiere un perfil emocional complejo que requiere análisis más profundo en contexto clínico."
    )
    
    # Nivel de intensidad
    if rasgo_valor > 60:
        nivel = "MARCADO"
        color = "🔴"
    elif rasgo_valor > 40:
        nivel = "MODERADO"
        color = "🟡"
    else:
        nivel = "LEVE"
        color = "🟢"
    
    return {
        "emocion_dominante": emo_nombre,
        "emocion_valor": emo_valor,
        "rasgo_dominante": rasgo_nombre,
        "rasgo_valor": rasgo_valor,
        "nivel_intensidad": nivel,
        "color_nivel": color,
        "analisis_especifico": analisis_especifico
    }

def interpretar_sd3_detallado(sd3):
    """Genera interpretación detallada de cada rasgo SD3"""
    interpretaciones = []
    
    for rasgo, valor in sd3.items():
        if valor > 60:
            nivel = "Alto"
            color = "🔴"
        elif valor > 40:
            nivel = "Moderado"
            color = "🟡"
        else:
            nivel = "Bajo"
            color = "🟢"
        
        descripciones = {
            "Maquiavelismo": {
                "Alto": "Tendencia marcada a manipulación estratégica, enfoque pragmático sobre ético, y habilidad para usar a otros como medios para fines.",
                "Moderado": "Balance entre pragmatismo y consideración ética. Puede ser estratégico sin caer en manipulación extrema.",
                "Bajo": "Enfoque honesto y directo en interacciones. Valora la transparencia sobre la estrategia."
            },
            "Narcisismo": {
                "Alto": "Autoestima inflada, búsqueda constante de admiración, y dificultad para aceptar críticas. El ego es central en las interacciones.",
                "Moderado": "Confianza personal equilibrada con cierta necesidad de validación externa. Autoestima saludable en general.",
                "Bajo": "Humildad genuina y consideración hacia otros. Baja necesidad de reconocimiento constante."
            },
            "Psicopatía": {
                "Alto": "Baja reactividad emocional, alta impulsividad, dificultad para empatía profunda. Procesamiento frío de situaciones sociales.",
                "Moderado": "Control emocional equilibrado con capacidad de conexión emocional moderada. Puede parecer distante en situaciones estresantes.",
                "Bajo": "Alta empatía, fuerte conexión emocional, y procesamiento afectivo profundo de experiencias."
            }
        }
        
        interpretaciones.append({
            "rasgo": rasgo,
            "valor": valor,
            "nivel": nivel,
            "color": color,
            "descripcion": descripciones[rasgo][nivel]
        })
    
    return interpretaciones

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

uploaded_file = st.file_uploader("Subí una imagen", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Imagen cargada', use_column_width=True)
    
    if st.button('🔍 Analizar imagen'):
        with st.spinner('Analizando microexpresiones...'):
            emotions, sd3 = analizar(image)
        
        st.success('✅ Análisis completado!')
        
        # ANÁLISIS COMPLETO
        analisis = analisis_personalidad_completo(emotions, sd3)
        
        st.markdown(f"""
        <div class="conclusion-box">
            <h2>🔬 Análisis Psicológico Completo</h2>
            <p class="emotion-dominant">
                Perfil Detectado: {analisis['emocion_dominante']} ({analisis['emocion_valor']*100:.1f}%) 
                + {analisis['rasgo_dominante']} ({analisis['rasgo_valor']:.1f}%)
            </p>
            <p><strong>Intensidad del rasgo:</strong> {analisis['color_nivel']} {analisis['nivel_intensidad']}</p>
            <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
            <h3>💡 Interpretación Específica:</h3>
            <p style="font-size: 1.1rem; line-height: 1.8;">{analisis['analisis_especifico']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráficos en columnas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Microexpresiones detectadas")
            df_emotions = pd.DataFrame({
                'Emoción': list(emotions.keys()),
                'Probabilidad': list(emotions.values())
            })
            st.bar_chart(df_emotions.set_index('Emoción'))
            
            st.write("**Valores detallados:**")
            for emo, val in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- **{emo}**: {val*100:.2f}%")
        
        with col2:
            st.subheader("🧠 Rasgos SD3 (Dark Triad)")
            df_sd3 = pd.DataFrame({
                'Rasgo': list(sd3.keys()),
                'Puntuación': list(sd3.values())
            })
            st.bar_chart(df_sd3.set_index('Rasgo'))
            
            st.write("**Valores:**")
            for rasgo, val in sorted(sd3.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- **{rasgo}**: {val:.2f}%")
        
        # INTERPRETACIÓN SD3 DETALLADA
        st.markdown("---")
        st.markdown("### 🔍 Interpretación Detallada de Rasgos SD3")
        
        interpretaciones_sd3 = interpretar_sd3_detallado(sd3)
        for interp in interpretaciones_sd3:
            st.markdown(f"""
            <div class="metric-box">
                <h4>{interp['color']} {interp['rasgo']}: {interp['nivel']} ({interp['valor']:.1f}%)</h4>
                <p style="margin-top: 0.5rem;">{interp['descripcion']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ADVERTENCIA
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Nota Importante:</strong> Este análisis es orientativo y basado en microexpresiones faciales. 
            No debe usarse como diagnóstico clínico ni para tomar decisiones importantes sobre personas. 
            Los rasgos SD3 son constructos psicológicos que existen en un continuo y requieren evaluación profesional.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👆 Sube una imagen para comenzar el análisis")
    
    # Información adicional
    st.markdown("---")
    st.markdown("### 🎯 ¿Qué hace esta aplicación?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 😊 Detección de Emociones")
        st.write("Identifica 7 emociones básicas: Alegría, Tristeza, Enojo, Sorpresa, Miedo, Disgusto y Neutral.")
    
    with col2:
        st.markdown("#### 🧠 Análisis SD3")
        st.write("Evalúa tres rasgos de personalidad oscura: Maquiavelismo, Narcisismo y Psicopatía.")
    
    with col3:
        st.markdown("#### 🔬 Interpretación Cruzada")
        st.write("Combina emociones y rasgos para generar un perfil psicológico específico.")
