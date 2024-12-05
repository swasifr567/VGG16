import os
import requests
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Function to download images
def download_images():
    # Check if images are already downloaded
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
def get_all_features(image_dir, model):
    features_list = []
    image_paths = os.listdir(image_dir)

    # Check if features have been extracted already
    if os.path.exists('features.pkl') and os.path.exists('image_paths.pkl'):
        with open('features.pkl', 'rb') as f:
            features_list = pickle.load(f)
        with open('image_paths.pkl', 'rb') as f:
            image_paths = pickle.load(f)
        print("Features already extracted, loading from saved files.")
    else:
        for img_name in image_paths:
            img_path = os.path.join(image_dir, img_name)
            features = extract_features(img_path, model)
            features_list.append(features)

        # Save features and image paths for future use
        with open('features.pkl', 'wb') as f:
            pickle.dump(features_list, f)
        with open('image_paths.pkl', 'wb') as f:
            pickle.dump(image_paths, f)
        print("Features extracted and saved.")

    return features_list, image_paths

# Function to display images side by side
def display_similar_images(image_paths):
    plt.figure(figsize=(15, 10))  # Adjust figure size for layout

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path)  # Open the image using PIL
        plt.subplot(1, len(image_paths), idx + 1)  # Create a subplot for each image
        plt.imshow(img)
        plt.axis('off')  # Turn off axes for cleaner visualization
        plt.title(f"Image {idx + 1}")

    plt.show()  # Display the plot

# Function to get similar images using cosine similarity
def get_similar_images(query_img_path, features_list, image_paths, model, top_n=5):
    query_features = extract_features(query_img_path, model)
    similarities = []

    for features in features_list:
        similarity = cosine_similarity(query_features.reshape(1, -1), features.reshape(1, -1))
        similarities.append(similarity[0][0])

    top_indices = np.argsort(similarities)[::-1][:top_n]
    similar_images = [os.path.join('Images', image_paths[i]) for i in top_indices]
    return similar_images

# Main function to run the app
def main():
    # Download images if not already present
    download_images()

    # Load the VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=vgg16_model.input, outputs=vgg16_model.output)

    # Get the features from images
    features_list, image_paths = get_all_features('Images', model)

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        img_path = os.path.join('Images', uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        similar_images = get_similar_images(img_path, features_list, image_paths, model)
        st.write("Top similar images:")
        
        # Display similar images side by side
        display_similar_images(similar_images)

if __name__ == "__main__":
    main()
