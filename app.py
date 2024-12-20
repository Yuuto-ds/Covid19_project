import torch
import streamlit as st
from PIL import Image
from utils import load_model, preprocess_image, generate_gradcam
import matplotlib.pyplot as plt

# Load the model
MODEL_PATH = "C:/Users/yuuto/Documents/Python_local/VM/Covid19_project/CNN_ResNet50_final_Torch_model.pth"
model = load_model(MODEL_PATH)

# Class labels
CLASS_LABELS = ['COVID', 'Non-COVID', 'Normal']

st.title("X-ray Image Classification with Grad-CAM")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Image Classification"])

# Home Page
if page == "Home":
    st.title("X-ray Image Classification App")
    st.write("""
    This Streamlit app allows you to classify chest X-ray images into three categories:
    - **COVID**
    - **Non-COVID**
    - **Normal**

    The deep learning model used for classification is based on a ResNet50 Convolutional Neural Network (CNN).
    """)

# Image Classification Page
elif page == "Image Classification":
    st.title("Upload an X-ray Image and Lung Mask for Classification")

    uploaded_file = st.file_uploader("Upload a PNG X-ray image", type=["png"])
    uploaded_mask = st.file_uploader("Upload a corresponding Lung Mask (PNG)", type=["png"])

    if uploaded_file is not None and uploaded_mask is not None:
        # Display the uploaded image and mask
        image = Image.open(uploaded_file).convert('RGB')
        mask = Image.open(uploaded_mask).convert('L')

        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        st.image(mask, caption="Uploaded Lung Mask", use_column_width=True)

        # Preprocess the image
        image_tensor = preprocess_image(image)

        # Predict the class and generate Grad-CAM
        with st.spinner("Classifying and generating Grad-CAM..."):
            try:
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1).squeeze()
                predicted_class = torch.argmax(probabilities).item()
                prediction_label = CLASS_LABELS[predicted_class]
                confidence = probabilities[predicted_class].item() * 100

                st.success(f"Prediction: **{prediction_label}** ({confidence:.2f}% confidence)")

                # Generate Grad-CAM
                gradcam_image = generate_gradcam(model, image_tensor, predicted_class, mask)

                # Display Grad-CAM using matplotlib
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                ax[0].imshow(image)
                ax[0].axis('off')
                ax[0].set_title("Original Image")

                ax[1].imshow(gradcam_image)
                ax[1].axis('off')
                ax[1].set_title("Grad-CAM Visualization")

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during model forward pass: {e}")
