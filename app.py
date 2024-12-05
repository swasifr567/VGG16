import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# Initialize Streamlit app
st.title("Image Similarity Finder")
st.write("Upload an image to find similar images from the dataset.")

# Load VGG16 model pre-trained on ImageNet
@st.cache_resource
def load_vgg16_model():
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=vgg16.input, outputs=vgg16.output)

model = load_vgg16_model()

# Function to download images from Google Sheets link
@st.cache_data
def download_images():
    # Use the raw URL for the CSV file on GitHub
    sheet_url = "https://raw.githubusercontent.com/swasifr567/VGG16/main/Data%20ID%20-%20Sheet1.csv"
    
    # Load the dataset from the GitHub raw link
    df = pd.read_csv(sheet_url)
    
    # Create the directory to store images if it doesn't exist
    os.makedirs("Images", exist_ok=True)
    
    # Download each image from the image_link in the dataset
    for index, row in df.iterrows():
        product_id = row['Product ID']
        image_url = row['image_link']
        image_path = os.path.join("Images", f"{product_id}.jpg")
        
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Downloaded: {product_id}.jpg")
            else:
                print(f"Failed to download {product_id}.jpg: Status {response.status_code}")
        except Exception as e:
            print(f"Error downloading {product_id}.jpg: {e}")

download_images()

# Extract features for all images
@st.cache_data
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

@st.cache_data
def get_all_features(image_dir):
    features_list = []
    image_paths = sorted(os.listdir(image_dir))
    for img_name in image_paths:
        img_path = os.path.join(image_dir, img_name)
        features = extract_features(img_path)
        features_list.append(features)
    return features_list, image_paths

features_list, image_paths = get_all_features('Images')

# Function to find similar images
def get_similar_images(query_img_path, features_list, image_paths, top_n=5):
    query_features = extract_features(query_img_path)
    similarities = [cosine_similarity(query_features.reshape(1, -1), feat.reshape(1, -1))[0][0] for feat in features_list]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    similar_images = [os.path.join('Images', image_paths[i]) for i in top_indices]
    return similar_images

# Upload and display query image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    with open("temp_query_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Get similar images
    similar_images = get_similar_images("temp_query_image.jpg", features_list, image_paths)

    # Display the similar images
    st.write("Top similar images:")
    for img_path in similar_images:
        img = Image.open(img_path)
        st.image(img, caption=os.path.basename(img_path), use_column_width=True)
