import streamlit as st
import numpy as np
import cv2
import os
from tensorflow import keras

# Define categories
CATEGORIES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load the model
model = keras.models.load_model(r'D:\arsha\DL\Plant Disease Project DL\plant_disease.h5')
# Recompile if necessary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess the image
def image(path):
    img = cv2.imread(path)
    if img is None:
        print("Failed to load image!")
        return None
    new_arr = cv2.resize(img, (100, 100))  # Resize to match model input
    new_arr = np.array(new_arr / 255.0)  # Normalize pixel values
    new_arr = new_arr.reshape(-1, 100, 100, 3)  # Reshape to add batch dimension
    return new_arr

# Streamlit UI setup
st.title("Plant Disease Prediction")
st.write("Upload an image of a plant leaf to identify the disease or if it's healthy.")

# File upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Save the uploaded image to a temporary path
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preprocess the image and make predictions
    img_array = image("uploaded_image.jpg")
    if img_array is not None:
        prediction = model.predict(img_array)
        
        # Find the index of the highest probability class
        predicted_class_index = prediction.argmax()
        predicted_class = CATEGORIES[predicted_class_index]
        
        st.write(f"Prediction: {predicted_class}")
    else:
        st.write("Error in processing the image!")
