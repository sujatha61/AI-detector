import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import open_clip

# ----------------------------
# Page config
st.set_page_config(page_title="AI Image Detector", layout="centered")

st.title("🖼️ Real vs AI-Generated Image Detection")

# ----------------------------
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
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
# Load models (cached)
@st.cache_resource
def load_models():
    # CNN
    cnn = SimpleCNN().to(device)
    cnn.load_state_dict(torch.load("cnn_model.pth", map_location=device))
    cnn.eval()

    # CLIP
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    clip_model = clip_model.to(device).eval()

    clip_classifier = nn.Linear(clip_model.visual.output_dim, 2)
    clip_classifier.load_state_dict(torch.load("clip_classifier.pth", map_location=device))
    clip_classifier = clip_classifier.to(device).eval()

    # DINO
    dino_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    dino_model.head = nn.Identity()
    dino_model = dino_model.to(device).eval()

    dino_classifier = nn.Linear(dino_model.num_features, 2)
    dino_classifier.load_state_dict(torch.load("dino_classifier.pth", map_location=device))
    dino_classifier = dino_classifier.to(device).eval()

    return cnn, clip_model, clip_classifier, dino_model, dino_classifier


cnn, clip_model, clip_classifier, dino_model, dino_classifier = load_models()

# ----------------------------
# Prediction
def predict_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_prob = F.softmax(cnn(img_tensor), dim=1)
        clip_prob = F.softmax(clip_classifier(clip_model.encode_image(img_tensor)), dim=1)
        dino_prob = F.softmax(dino_classifier(dino_model(img_tensor)), dim=1)

        ensemble = (cnn_prob + clip_prob + dino_prob) / 3
        pred = torch.argmax(ensemble, dim=1).item()
        conf = ensemble[0][pred].item()

    return pred, conf

# ----------------------------
# Upload only (NO CAMERA)
uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    pred, conf = predict_image(image)

    if pred == 1:
        st.error("🚨 AI-Generated Image")
    else:
        st.success("✅ Real Image")

    st.info(f"Confidence: {conf*100:.2f}%")
