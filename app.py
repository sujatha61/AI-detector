import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import os

# ----------------------------
st.title("🖼️ AI vs Real Image Detector")

# ----------------------------
# Device (force CPU)
device = "cpu"

# ----------------------------
# Model path
model_path = "cnn_model.pth"

# ----------------------------
# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128,2)

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

# ----------------------------
# Load model (cached)
@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

# ----------------------------
# Prediction
def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    return "AI-Generated" if pred.item() == 1 else "Real"

# ----------------------------
# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    result = predict(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if result == "AI-Generated":
        st.error("AI-Generated Image")
    else:
        st.success("Real Image")
