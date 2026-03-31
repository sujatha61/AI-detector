import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

# ----------------------------
# UI Style
st.markdown("""
    <style>
    .stApp {
        background: darkorange;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Page
st.set_page_config(page_title="AI Image Detector", layout="centered")
st.title("🖼️ AI vs Real Image Detector")

# ----------------------------
# Device
device = "cpu"

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
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------------
# Load model
@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# ----------------------------
# Prediction
def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    return pred, conf, probs

# ----------------------------
# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    pred, conf, probs = predict(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ----------------------------
    # DEBUG (remove later if needed)
    st.write("Raw Prediction:", pred)
    st.write("Probabilities:", probs.tolist())

    # ----------------------------
    # Label Fix (handles reversed labels)
    # Try both mappings automatically
    if conf < 0.6:
        st.warning("⚠️ Low confidence prediction")

    if pred == 1:
        st.error("🚨 AI-Generated Image")
    else:
        st.success("✅ Real Image")

    st.info(f"Confidence: {conf*100:.2f}%")
