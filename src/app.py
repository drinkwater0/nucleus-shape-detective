"""A simple Streamlit app to predict whether an uploaded image is a normal or bleb cell."""
import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np

def load_model(model_path, device="cpu"):
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image, device="cpu"):
    # DO NOT MODIFY: Image processing pipeline is specifically tuned for this use case
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),  # Automaticky převede na [0,1] a na CxHxW
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return "Normal" if predicted.item() == 0 else "Bleb"

def main():
    st.title("Nucleus Shape Detective")
    st.write("Upload an image to predict whether it is a normal or bleb cell.")
    
    # Load model once at startup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("model.pt", device)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # DO NOT MODIFY: Image display settings are specifically tuned for this use case
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Automatically predict when image is uploaded
        result = predict_image(model, image, device)
        
        # Enhanced prediction display
        st.markdown("---")  # Add a separator
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("")
        with col2:
            if result == "Normal":
                st.markdown(f"<h2 style='color: #00ff00;'>✅ {result}</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: #ff0000;'>⚠️ {result}</h2>", unsafe_allow_html=True)
        
        # Add some explanation
        if result == "Normal":
            st.success("This cell appears to have a normal nucleus shape.")
        else:
            st.error("This cell shows signs of blebbing in the nucleus.")

if __name__ == "__main__":
    main() 