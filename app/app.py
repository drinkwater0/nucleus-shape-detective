"""Streamlit front‑end for drag‑and‑drop nucleus classification."""
import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np
import os

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
        probs = torch.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
    return "Normal" if predicted.item() == 0 else "Bleb", float(probs.max())

def load_example_images(normal_index=0, bleb_index=0):
    """Load example images from the data directory."""
    try:
        # Get images from normal and bleb directories
        normal_dir = "data/normal"
        bleb_dir = "data/bleb"
        
        normal_images = sorted([f for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        bleb_images = sorted([f for f in os.listdir(bleb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if normal_images and bleb_images:
            # Use modulo to wrap around if index is out of range
            normal_idx = normal_index % len(normal_images)
            bleb_idx = bleb_index % len(bleb_images)
            
            normal_img = Image.open(os.path.join(normal_dir, normal_images[normal_idx])).convert("RGB")
            bleb_img = Image.open(os.path.join(bleb_dir, bleb_images[bleb_idx])).convert("RGB")
            return normal_img, bleb_img
    except Exception as e:
        st.warning(f"Could not load example images: {str(e)}")
    return None, None

def main():
    st.set_page_config(page_title="Nucleus‑Shape Detective")
    st.title("Nucleus‑Shape Detective")
    st.write("Upload an image to predict whether it is a normal or bleb cell.")
    
    # Load model once at startup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("model.pt", device)
    
    # Show example images
    st.markdown("### Example Images")
    
    # Add sliders to select example images
    col1, col2 = st.columns(2)
    with col1:
        normal_index = st.slider("Select Normal Example", 0, 10, 0)
    with col2:
        bleb_index = st.slider("Select Bleb Example", 0, 10, 0)
    
    normal_img, bleb_img = load_example_images(normal_index, bleb_index)
    if normal_img and bleb_img:
        col1, col2 = st.columns(2)
        with col1:
            st.image(normal_img, caption="Example of Normal Nucleus", use_container_width=True)
            st.markdown("<p style='text-align: center; color: #00ff00;'>✅ Normal</p>", unsafe_allow_html=True)
        with col2:
            st.image(bleb_img, caption="Example of Blebbed Nucleus", use_container_width=True)
            st.markdown("<p style='text-align: center; color: #ff0000;'>⚠️ Blebbed</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Upload Your Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # DO NOT MODIFY: Image display settings are specifically tuned for this use case
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Automatically predict when image is uploaded
        result, confidence = predict_image(model, image, device)
        
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
        
        # Display confidence
        st.markdown(f"**Confidence:** {confidence:.1%}")
        
        # Add some explanation
        if result == "Normal":
            st.success("This cell appears to have a normal nucleus shape.")
        else:
            st.error("This cell shows signs of blebbing in the nucleus.")

if __name__ == "__main__":
    main()
