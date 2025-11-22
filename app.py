import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Cargar Modelo
# ---------------------------

# Descargar automáticamente tu archivo .pth desde tu repo
ckpt_path = hf_hub_download(
    repo_id="psicoit/DarkLens-model",
    filename="exp_retrained_FER2013.pth"
)

# Definir modelo
class MicroExpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 48 * 48, 7)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = MicroExpNet().to(device)

# Cargar pesos
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

labels = ["Enojo", "Asco", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Neutral"]

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# ---------------------------
# Función de predicción
# ---------------------------
def analizar(img):
    img = Image.fromarray(img).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out).item()

    return labels[pred]


# ---------------------------
# Interfaz (estética violeta futurista)
# ---------------------------

custom_css = """
body {background: linear-gradient(135deg, #3a0066, #8a00c2);}
.gradio-container {color: white !important;}
button {background: #b300ff !important; color:white !important;}
"""

demo = gr.Interface(
    fn=analizar,
    inputs=gr.Image(type="numpy", label="Subí una imagen"),
    outputs=gr.Text(label="Resultado"),
    title="DarkLens",
    description="Analizador de microexpresiones - Modelo Personalizado",
    css=custom_css
)

demo.launch()
