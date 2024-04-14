import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the trained model
model = load_model('Brain Tumor detection.g9')

# Function to make predictions
def predict(image):
    img = Image.open(image)
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    return res

# Streamlit App
def main():
    st.title('Brain Tumor Detection')

    st.write("""
    Upload an MRI image for brain tumor prediction.
    """)

    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        prediction = predict(uploaded_file)
        if prediction == [[0.]]:
            st.write("No tumor detected.")
        else:
            st.write("Tumor detected.")

if __name__ == '__main__':
    main()
