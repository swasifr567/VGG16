import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Streamlit app
st.title("Image Similarity Finder")
st.write("Upload an image to find similar images from the dataset.")

# Create the directory to store images
image_dir = "Images"
os.makedirs(image_dir, exist_ok=True)

# Load the dataset from Google Sheets
sheet_url = "https://docs.google.com/spreadsheets/d/121aV7BjJqCRlFcVegbbhI1Zmt67wG61ayRiFtDnafKY/export?format=csv&gid=0"
df = pd.read_csv(sheet_url)

# Download images if not already downloaded
if st.button("Download Images"):
    st.write("Downloading images...")
    for index, row in df.iterrows():
        product_id = row['Product ID']
        image_url = row['image_link']
        image_path = os.path.join(image_dir, f"{product_id}.jpg")

        if not os.path.exists(image_path):
            try:
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    st.write(f"Downloaded: {product_id}.jpg")
                else:
                    st.write(f"Failed to download {product_id}.jpg: Status {response.status_code}")
            except Exception as e:
                st.write(f"Error downloading {product_id}.jpg: {e}")
    st.write("Download complete!")

# Load the VGG16 model pre-trained on ImageNet
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=vgg16_model.input, outputs=vgg16_model.output)

# Feature extraction function
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Extract features for all images in the dataset
@st.cache
def get_all_features(image_dir):
    features_list = []
    image_paths = os.listdir(image_dir)
    for img_name in image_paths:
        img_path = os.path.join(image_dir, img_name)
        features = extract_features(img_path)
        features_list.append(features)
    return features_list, image_paths

if st.button("Extract Features"):
    features_list, image_paths = get_all_features(image_dir)
    st.write("Feature extraction complete!")

# Similarity calculation function
def get_similar_images(query_img_path, features_list, image_paths, top_n=5):
    query_features = extract_features(query_img_path)
    similarities = [
        cosine_similarity(query_features.reshape(1, -1), feat.reshape(1, -1))[0][0]
        for feat in features_list
    ]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [os.path.join(image_dir, image_paths[i]) for i in top_indices]

# Upload an image for querying
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    query_image_path = "query_image.jpg"
    with open(query_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Find and display similar images
    if st.button("Find Similar Images"):
        similar_images = get_similar_images(query_image_path, features_list, image_paths)
        st.write("Top similar images:")
        for img_path in similar_images:
            img = Image.open(img_path)
            st.image(img, caption=img_path, use_column_width=True)
