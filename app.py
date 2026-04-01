import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
import smtplib

# ----------------------------
# EMAIL CONFIG (USE APP PASSWORD)
SENDER_EMAIL = "sujatha6153@gamil.com"
APP_PASSWORD = "mkfr lmoe rwyw rjtz"  # 🔴 Use Gmail App Password


# UI Style
st.markdown("""
    <style>
    .stApp {
        background: darkorange;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# SEND OTP FUNCTION
def send_otp(receiver_email):
    otp = str(random.randint(100000, 999999))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)

        message = f"Subject: OTP Verification\n\nYour OTP is: {otp}"
        server.sendmail(SENDER_EMAIL, receiver_email, message)
        server.quit()

        return otp
    except:
        return None

# ----------------------------
# UI CONFIG
st.set_page_config(page_title="AI Detector", layout="centered")

# ----------------------------
# SESSION STATE
if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ----------------------------
# LOGIN PAGE
if not st.session_state.otp_sent:
    st.title("🔐 Login")

    email = st.text_input("Enter Email")

    if st.button("Send OTP"):
        otp = send_otp(email)

        if otp:
            st.session_state.otp = otp
            st.session_state.email = email
            st.session_state.otp_sent = True
            st.success("OTP sent ✅")
        else:
            st.error("Failed to send OTP ❌")

# ----------------------------
# OTP VERIFY
elif not st.session_state.logged_in:
    st.title("📩 Enter OTP")

    user_otp = st.text_input("Enter OTP")

    if st.button("Verify"):
        if user_otp == st.session_state.otp:
            st.session_state.logged_in = True
            st.success("Login successful ✅")
        else:
            st.error("Wrong OTP ❌")

# ----------------------------
# MAIN APP
if st.session_state.logged_in:

    st.title("🖼️ AI vs Real Image Detector")

    device = "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ----------------------------
    # MODEL
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

    @st.cache_resource
    def load_model():
        model = SimpleCNN()
        model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
        model.eval()
        return model

    model = load_model()

    # ----------------------------
    # PREDICT
    def predict(image):
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            probs = F.softmax(model(img), dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        return pred, conf

    # ----------------------------
    # UPLOAD
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_container_width=True)

        pred, conf = predict(image)

        # Confidence check
        if conf < 0.6:
            st.warning("⚠️ Low confidence")

        if pred == 1:
            st.error(f"🚨 AI Generated ({conf*100:.2f}%)")
        else:
            st.success(f"✅ Real Image ({conf*100:.2f}%)")

    # ----------------------------
    # LOGOUT
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.otp_sent = False
