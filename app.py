import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# ----------------------------
# CONFIG (ADD YOUR DETAILS)
SENDGRID_API_KEY = "XUS7FBA8JNH62DU415MGXFQJ"
SENDER_EMAIL = "sujatha6153@gmail.com"

# ----------------------------
# UI Style
st.set_page_config(page_title="AI Image Detector", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background: darkorange;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# OTP FUNCTIONS
def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(to_email, otp):
    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=to_email,
        subject="Your OTP Code",
        html_content=f"<strong>Your OTP is: {otp}</strong>"
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        return True
    except Exception as e:
        print(e)
        return False

# ----------------------------
# SESSION
if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ----------------------------
# LOGIN FLOW
if not st.session_state.otp_sent:
    st.title("🔐 Login with Email")
    email = st.text_input("Enter your Email")

    if st.button("Send OTP"):
        otp = generate_otp()
        st.session_state.otp = otp
        st.session_state.email = email

        if send_otp_email(email, otp):
            st.session_state.otp_sent = True
            st.success("OTP sent successfully ✅")
        else:
            st.error("Failed to send OTP ❌")

elif not st.session_state.logged_in:
    st.title("📩 Enter OTP")
    otp_input = st.text_input("Enter OTP")

    if st.button("Verify"):
        if otp_input == st.session_state.otp:
            st.session_state.logged_in = True
        else:
            st.error("Invalid OTP ❌")

# ----------------------------
# MAIN APP
if st.session_state.logged_in:

    st.title("🖼️ AI vs Real Image Detector")

    device = "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # CNN MODEL
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

    # LOAD MODEL
    @st.cache_resource
    def load_model():
        model = SimpleCNN()
        model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
        model.eval()
        return model

    model = load_model()

    # PREDICT
    def predict(image):
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(img)
            probs = F.softmax(logits, dim=1)

            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        return pred, conf

    # UPLOAD
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        pred, conf = predict(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        # OUTPUT FIX
        if conf < 0.6:
            st.warning("⚠️ Low confidence")

        if pred == 1:
            st.error(f"🚨 AI Generated ({conf*100:.2f}%)")
        else:
            st.success(f"✅ Real Image ({conf*100:.2f}%)")

    # LOGOUT
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.otp_sent = False
