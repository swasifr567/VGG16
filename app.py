import os
import requests
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Function to download images if they are not already downloaded
@st.cache_data
def download_images():
    if not os.path.exists('Images') or len(os.listdir('Images')) == 0:
        sheet_url = "https://raw.githubusercontent.com/swasifr567/VGG16/main/Data%20ID%20-%20Sheet1.csv"
        df = pd.read_csv(sheet_url)

        os.makedirs("Images", exist_ok=True)

        for index, row in df.iterrows():
            product_id = row['Product ID']
            image_url = row['image_link']
            image_path = os.path.join("Images", f"{product_id}.jpg")

            try:
                if not os.path.exists(image_path):
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        with open(image_path, 'wb') as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)
                        print(f"Downloaded: {product_id}.jpg")
                    else:
                        print(f"Failed to download {product_id}.jpg: Status {response.status_code}")
                else:
                    print(f"Image {product_id}.jpg already exists, skipping download.")
            except Exception as e:
                print(f"Error downloading {product_id}.jpg: {e}")
    else:
        print("Images already downloaded.")

# Load the VGG16 model
@st.cache_resource
def load_model():
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=vgg16_model.input, outputs=vgg16_model.output)
    return model

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # resize image to 224x224
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for VGG16
    features = model.predict(img_array)  # Get Features
    features = features.flatten()  # Flatten the features to 1D array
    return features

# Function to extract features for all images
@st.cache_data
def get_all_features(image_dir, model):
    features_list = []
    image_paths = os.listdir(image_dir)
    for img_name in image_paths:
        img_path = os.path.join(image_dir, img_name)
        features = extract_features(img_path, model)
        features_list.append(features)

    # Save features and image paths for future use
    with open('features.pkl', 'wb') as f:
        pickle.dump(features_list, f)
    with open('image_paths.pkl', 'wb') as f:
        pickle.dump(image_paths, f)

    return features_list, image_paths

# Function to get similar images
def get_similar_images(query_img_path, features_list, image_paths, model, top_n=5):
    query_features = extract_features(query_img_path, model)
    similarities = []
    for features in features_list:
        similarity = cosine_similarity(query_features.reshape(1, -1), features.reshape(1, -1))
        similarities.append(similarity[0][0])
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    similar_images = [image_paths[i] for i in top_indices]
    return similar_images

# Streamlit UI for image input and display
def display_similar_images(query_image, features_list, image_paths, model):
    st.image(query_image, caption="Query Image", use_column_width=True)

    # Get the top 5 similar images
    similar_images = get_similar_images(query_image, features_list, image_paths, model)

    # Display similar images
    st.write("Top Similar Images:")

    cols = st.columns(len(similar_images))  # Create columns for each similar image
    for i, img_name in enumerate(similar_images):
        img_path = os.path.join('Images', img_name)
        img = Image.open(img_path)
        cols[i].image(img, caption=f"Image {i+1}: {img_name}", use_column_width=True)

# Main function for Streamlit app
def main():
    st.title("Similar Image Search")

    # Step 1: Download images
    download_images()

    # Step 2: Load the VGG16 model
    model = load_model()

    # Step 3: Extract features if not already extracted
    if not os.path.exists('features.pkl') or not os.path.exists('image_paths.pkl'):
        features_list, image_paths = get_all_features('Images', model)
    else:
        with open('features.pkl', 'rb') as f:
            features_list = pickle.load(f)
        with open('image_paths.pkl', 'rb') as f:
            image_paths = pickle.load(f)

    # Step 4: User uploads query image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Step 5: Display query image and similar images
        query_image = Image.open(uploaded_image)
        display_similar_images(uploaded_image, features_list, image_paths, model)

if __name__ == "__main__":
    main()
