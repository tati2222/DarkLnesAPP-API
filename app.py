import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# --------------------------
#   CARGA DEL MODELO
# --------------------------

class MicroExpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 7)

    def forward(self, x):
        return self.model(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MicroExpNet()
state = torch.load("microexp_retrained_FER2013.pth", map_location=device)
new_state = {}

for k, v in state.items():
    nk = k.replace("model.", "")
    new_state[nk] = v

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
# RELACIÓN CON SD3
# --------------------------

def compute_sd3(emotions):
    # Valores ficticios, tú puedes afinarlos luego
    maqu = emotions["Enojo"] * 0.6 + emotions["Disgusto"] * 0.4
    narc = emotions["Alegría"] * 0.5 + emotions["Neutral"] * 0.5
    psic = emotions["Miedo"] * 0.7 + emotions["Sorpresa"] * 0.3

    return {
        "Maquiavelismo": round(maqu * 100, 2),
        "Narcisismo": round(narc * 100, 2),
        "Psicopatía": round(psic * 100, 2)
    }

# --------------------------
# FUNCIÓN PRINCIPAL
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
# INTERFAZ GRADIO - DARKLENS
# --------------------------

css = """
body {
    background: radial-gradient(circle at center, #3a0066, #14001f);
    color: white !important;
}
.gradio-container {
    background: transparent !important;
    color: white !important;
}
label, .label, .title, h1, h2, h3, p, span {
    color: white !important;
}
button {
    background: #6a0dad !important;
    color: white !important;
    border-radius: 8px !important;
}
"""

with gr.Blocks(css=css, title="DarkLens") as app:
    gr.Markdown("<h1 style='text-align:center;'>🟣 DarkLens — Detector de Microexpresiones</h1>")

    with gr.Row():
        img_input = gr.Image(type="numpy", label="Subí una imagen")
        btn = gr.Button("🔍 Analizar imagen")

    with gr.Row():
        emociones_out = gr.BarPlot(
            label="Microexpresiones detectadas",
            x="labels",
            y="values",
        )

        sd3_out = gr.BarPlot(
            label="Rasgos SD3",
            x="labels",
            y="values",
        )

    def process(img):
        emotions, sd3 = analizar(img)
        emo_labels = list(emotions.keys())
        emo_values = list(emotions.values())

        sd3_labels = list(sd3.keys())
        sd3_values = list(sd3.values())

        return {
            emociones_out: {"labels": emo_labels, "values": emo_values},
            sd3_out: {"labels": sd3_labels, "values": sd3_values}
        }

    btn.click(process, inputs=[img_input], outputs=[emociones_out, sd3_out])

app.launch()
