"""Streamlit front‑end for drag‑and‑drop nucleus classification."""
import streamlit as st, torch, torchvision
from PIL import Image
from torchvision import transforms as T

st.set_page_config(page_title="Nucleus‑Shape Detective")
st.title("Nucleus‑Shape Detective")

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

uploaded = st.file_uploader("Upload a nucleus image", type=["png", "jpg", "tif", "tiff"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Input image", width=256)
    probs = torch.softmax(model(preprocess(img).unsqueeze(0)), dim=1)[0]
    label = "bleb" if probs[1] > probs[0] else "normal"
    st.write(f"**Prediction:** {label} (confidence {float(probs.max()):.2f})")
