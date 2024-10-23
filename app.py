import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the trained model
model = load_model('Brain Tumor detection.g9')

# Function to make predictions
def predict(image):
    img = Image.open(image)
    img = img.resize((128, 128))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize the image
    input_img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    res = model.predict(input_img)
    return res

# Streamlit App
def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
        body {
            background-color: #f0f2f5;
        }
        .title {
            text-align: center;
            font-size: 2.5em;
            color: #1a1a1a;
        }
        .prediction {
            text-align: center;
            font-size: 2em;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        .image-container {
            margin: 20px 0;
        }
        .upload-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .upload-button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Brain Tumor Detection")

    st.write("""
    Upload an MRI image for brain tumor prediction.
    """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Make prediction
        prediction = predict(uploaded_file)
        if prediction[0][0] < 0.5:  # Adjust threshold based on your model's output
            st.markdown("<div class='prediction' style='color: green;'>No tumor detected.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='prediction' style='color: red;'>Tumor detected!</div>", unsafe_allow_html=True)

        st.write("By shanmuk")

if __name__ == '__main__':
    main()
