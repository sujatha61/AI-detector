import streamlit as st
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import open_clip
import os

# ----------------------------
# UI Styling
st.set_page_config(page_title="AI Image Detector", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #f5a623;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🖼️ Real vs AI-Generated Image Detection")

# ----------------------------
# Paths
base_path = r"C:\Users\sujat\AI_Image_Detection"
cnn_path = os.path.join(base_path, "cnn_model.pth")
clip_path = os.path.join(base_path, "clip_classifier.pth")
dino_path = os.path.join(base_path, "dino_classifier.pth")

# ----------------------------
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Image Enhancement (for webcam images)
def enhance_image(image):
    image = ImageEnhance.Brightness(image).enhance(1.5)
    image = ImageEnhance.Contrast(image).enhance(1.3)
    return image

# ----------------------------
# Transforms (CNN + DINO)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------
# Load CNN
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

cnn = SimpleCNN().to(device)
cnn.load_state_dict(torch.load(cnn_path, map_location=device))
cnn.eval()

# ----------------------------
# Load CLIP (FIXED)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model = clip_model.to(device).eval()

clip_classifier = nn.Linear(clip_model.visual.output_dim, 2)
clip_classifier.load_state_dict(torch.load(clip_path, map_location=device))
clip_classifier = clip_classifier.to(device).eval()

# ----------------------------
# Load DINOv2
dino_model = timm.create_model('vit_base_patch16_224', pretrained=True)
dino_model.head = nn.Identity()
dino_model = dino_model.to(device).eval()

dino_classifier = nn.Linear(dino_model.num_features, 2)
dino_classifier.load_state_dict(torch.load(dino_path, map_location=device))
dino_classifier = dino_classifier.to(device).eval()

# ----------------------------
# Prediction Function
def predict_image(image: Image.Image):
    labels = {0: "Real", 1: "AI-Generated"}

    # Enhance image
    image = enhance_image(image)

    # Prepare inputs
    cnn_input = transform(image).unsqueeze(0).to(device)
    clip_input = clip_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # CNN
        cnn_logits = cnn(cnn_input)
        cnn_prob = F.softmax(cnn_logits, dim=1)

        # CLIP
        clip_features = clip_model.encode_image(clip_input)
        clip_logits = clip_classifier(clip_features)
        clip_prob = F.softmax(clip_logits, dim=1)

        # DINO
        dino_logits = dino_classifier(dino_model(cnn_input))
        dino_prob = F.softmax(dino_logits, dim=1)

        # ----------------------------
        # Weighted Ensemble (FIXED)
        ensemble_prob = (
            0.4 * cnn_prob +
            0.4 * clip_prob +
            0.2 * dino_prob
        )

        final_pred = torch.argmax(ensemble_prob, dim=1).item()
        final_conf = ensemble_prob[0][final_pred].item()

        # ----------------------------
        # Confidence Threshold (NEW)
        
        final_label = labels[final_pred]

    return {
        "final_label": final_label,
        "final_conf": final_conf,
        "cnn_conf": cnn_prob.max().item(),
        "clip_conf": clip_prob.max().item(),
        "dino_conf": dino_prob.max().item()
    }

# ----------------------------
# Upload UI
uploaded_file = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        result = predict_image(image)

        st.subheader("Result")

        if result["final_label"] == "AI-Generated":
            st.error("AI-Generated")
        else:
            st.success("Real")

            st.info(f"Confidence: {result['final_conf']*100:.2f}%")

        if result["final_conf"] < 0.60:
            st.warning("Low confidence prediction")

        st.info(f"Confidence: {result['final_conf']*100:.2f}%")

        st.subheader("🔍 Model Confidence")
        st.write(f"CNN: {result['cnn_conf']*100:.2f}%")
        st.write(f"CLIP: {result['clip_conf']*100:.2f}%")
        st.write(f"DINO: {result['dino_conf']*100:.2f}%")

        st.subheader("🧠 Explanation")
        if result["final_label"] == "AI-Generated":
            st.write("Patterns suggest synthetic generation (textures, artifacts, consistency).")
        elif result["final_label"] == "Real":
            st.write("Image shows natural variations typical of real-world photography.")
        else:
            st.write("The model is unsure. Try a clearer or better-lit image.")

    with col2:
            st.image(image, caption="Uploaded Image", width="stretch")
