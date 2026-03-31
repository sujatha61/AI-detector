import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import open_clip
import os


import streamlit as st

st.markdown("""
    <style>
    .stApp {
        background: darkorange;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)



# ----------------------------
# Paths to saved models
cnn_path = "cnn_model.pth"
clip_path = "clip_classifier.pth"
dino_path = "dino_classifier.pth"

# ----------------------------
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Image transforms
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
# Load CLIP
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
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
# Cross-Entropy Loss
criterion = nn.CrossEntropyLoss()

# ----------------------------
# Prediction function with Cross-Entropy Loss
def predict_image(image: Image.Image, true_label=None):
    img_tensor = transform(image).unsqueeze(0).to(device)
    labels = {0: "Real", 1: "AI-Generated"}

    with torch.no_grad():
        # CNN
        cnn_logits = cnn(img_tensor)
        cnn_prob = F.softmax(cnn_logits, dim=1)
        cnn_pred = torch.argmax(cnn_prob, dim=1).item()
        cnn_conf = cnn_prob[0][cnn_pred].item()

        # CLIP
        clip_input = clip_preprocess(image).unsqueeze(0).to(device)
        clip_features = clip_model.encode_image(clip_input).float()
        clip_logits = clip_classifier(clip_features)
        clip_prob = F.softmax(clip_logits, dim=1)
        clip_pred = torch.argmax(clip_prob, dim=1).item()
        clip_conf = clip_prob[0][clip_pred].item()

        # DINOv2
        dino_logits = dino_classifier(dino_model(img_tensor))
        dino_prob = F.softmax(dino_logits, dim=1)
        dino_pred = torch.argmax(dino_prob, dim=1).item()
        dino_conf = dino_prob[0][dino_pred].item()

        # Ensemble
        ensemble_prob = (cnn_prob + clip_prob + dino_prob) / 3
        final_pred = torch.argmax(ensemble_prob, dim=1).item()
        final_label = labels[final_pred]
        final_conf = ensemble_prob[0][final_pred].item()

        # ----------------------------
        # Compute Cross-Entropy Loss if true_label is given
        cnn_loss = clip_loss = dino_loss = avg_loss = None
        if true_label is not None:
            true_tensor = torch.tensor([true_label]).to(device)
            cnn_loss = criterion(cnn_logits, true_tensor).item()
            clip_loss = criterion(clip_logits, true_tensor).item()
            dino_loss = criterion(dino_logits, true_tensor).item()
            avg_loss = (cnn_loss + clip_loss + dino_loss) / 3

    return {
        "CNN": (labels[cnn_pred], cnn_conf, cnn_loss),
        "CLIP": (labels[clip_pred], clip_conf, clip_loss),
        "DINOv2": (labels[dino_pred], dino_conf, dino_loss),
        "FINAL": (final_label, final_conf, avg_loss)
    }

# ----------------------------
# Streamlit App
st.title("🖼️ Real vs AI-Generated Image Detection")

st.subheader("📤 Choose Input Method")

option = st.radio(
    "Select input type:",
    ["Upload Image", "Use Camera"]
)

image = None

# ----------------------------
# Option 1: File Upload
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# ----------------------------
# Option 2: Camera Capture
elif option == "Use Camera":
    camera_file = st.camera_input("Take a photo")
    
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    preds = predict_image(image, true_label=true_label)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📊 Detection Result")
        final_label, final_conf, avg_loss = preds["FINAL"]

        if final_label == "AI-Generated":
            st.error(f"Prediction: {final_label}")
        else:
            st.success(f"Prediction: {final_label}")

        st.info(f"Confidence Score: {final_conf*100:.2f}%")

        st.markdown("### 🔍 Model Predictions")
        st.write(f"CNN: {preds['CNN'][0]} ({preds['CNN'][1]*100:.2f}%)")
        st.write(f"CLIP: {preds['CLIP'][0]} ({preds['CLIP'][1]*100:.2f}%)")
        st.write(f"DINOv2: {preds['DINOv2'][0]} ({preds['DINOv2'][1]*100:.2f}%)")

        # Display Cross-Entropy Loss if available
        if avg_loss is not None:
            st.markdown("### 🔢 Cross-Entropy Loss (per model)")
            st.write(f"CNN Loss: {preds['CNN'][2]:.4f}")
            st.write(f"CLIP Loss: {preds['CLIP'][2]:.4f}")
            st.write(f"DINOv2 Loss: {preds['DINOv2'][2]:.4f}")
            st.write(f"Average Loss: {avg_loss:.4f}")

        st.markdown("### 🧠 Explanation")
        if final_label == "AI-Generated":
            st.write(
                "The uploaded image exhibits characteristics commonly associated "
                "with AI-generated content. Texture consistency and feature patterns "
                "suggest synthetic generation."
            )
        else:
            st.write(
                "The uploaded image appears to be a natural photograph. "
                "Visual patterns and texture distribution align with real-world capture."
            )

    with col2:
        st.image(image, caption="Uploaded Image", width=400)  # set width as needed