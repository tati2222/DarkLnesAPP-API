from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn
import os

# -----------------------------------------------------
# MODELO (MISMO QUE USÁS EN STREAMLIT)
# -----------------------------------------------------
class MicroExpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.model(x)

# Cargar modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Usando dispositivo: {device}")

model = MicroExpNet()

try:
    # Intenta cargar el modelo - AQUÍ ESTÁ LA CLAVE
    state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
    first_key = list(state.keys())[0]

    # Ajustar claves igual que en tu Streamlit
    if first_key.startswith("model.model."):
        new_state = {k.replace("model.", "", 1): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=True)
    elif first_key.startswith("model."):
        new_state = {k.replace("model.", ""): v for k, v in state.items()}
        model.model.load_state_dict(new_state, strict=True)
    else:
        model.model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    print("✅ Modelo cargado exitosamente!")
    
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    # Modelo de emergencia
    model = None

# -----------------------------------------------------
# PREPROCESAMIENTO
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels = ["Alegría", "Tristeza", "Enojo", "Sorpresa", "Miedo", "Disgusto", "Neutral"]

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
# FASTAPI + CORS
# -----------------------------------------------------
app = FastAPI(title="DarkLnes Microexpressions API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check ESSENCIAL para Render
@app.get("/")
async def root():
    return {
        "message": "DarkLnes Microexpressions API", 
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# -----------------------------------------------------
# ENDPOINT /run/predict
# -----------------------------------------------------
@app.post("/run/predict")
async def run_predict(file: UploadFile = File(...)):
    try:
        print(f"📨 Recibiendo archivo: {file.filename}")
        
        if not model:
            raise HTTPException(status_code=500, detail="Modelo no cargado")
        
        # Leer y procesar imagen
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        print(f"📷 Imagen cargada: {img.size}")
        
        # Preprocesar
        tensor = transform(img).unsqueeze(0).to(device)
        print("🔬 Procesando con el modelo...")
        
        # Predecir
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        
        print(f"📊 Predicciones: {probs}")
        
        # Formatear resultados
        emotions = {labels[i]: float(probs[i]) for i in range(7)}
        sd3 = compute_sd3(emotions)
        
        print(f"🎭 SD3 calculado: {sd3}")
        
        return {
            "status": "ok",
            "emociones": emotions,
            "sd3": sd3,
            "modelo_utilizado": "EfficientNet-B0 entrenado en FER2013"
        }
        
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# Para Render - usa el puerto que ellos proveen
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
