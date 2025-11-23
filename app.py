import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os

# --------------------------
#   CARGA DEL MODELO
# --------------------------

class MicroExpNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Carga del modelo SIN internet (Render safe)
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.model(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MicroExpNet()

# Cargar pesos entrenados
state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
new_state = {k.replace("model.", ""): v for k, v in state.items()}

model.load_state_dict(new_state, strict=False)
model.to(device)
model.eval()

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

def analizar(img):
    image = Image.fromarray(img)
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

css = """
body { background: radial-gradient(circle at center, #3a0066, #14001f); color: white !important; }
.gradio-container { background: transparent !important; color: white !important; }
label, .label, .title, h1, h2, h3, p, span { color: white !important; }
button { background: #6a0dad !important; color: white !important; border-radius: 8px !important; }
"""

with gr.Blocks(css=css, title="DarkLens") as app:
    gr.Markdown("<h1 style='text-align:center;'>🟣 DarkLens — Detector de Microexpresiones</h1>")

    with gr.Row():
        img_input = gr.Image(type="numpy", label="Subí una imagen")
        btn = gr.Button("🔍 Analizar imagen")

    with gr.Row():
        emociones_out = gr.BarPlot(label="Microexpresiones detectadas", x="labels", y="values")
        sd3_out = gr.BarPlot(label="Rasgos SD3", x="labels", y="values")

    def process(img):
        emotions, sd3 = analizar(img)

        return {
            emociones_out: {"labels": list(emotions.keys()), "values": list(emotions.values())},
            sd3_out: {"labels": list(sd3.keys()), "values": list(sd3.values())}
        }

    btn.click(process, inputs=[img_input], outputs=[emociones_out, sd3_out])

# --------------------------
# RENDER
# --------------------------

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=PORT)
