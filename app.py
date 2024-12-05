import streamlit as st
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Initialize VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Function to get all features from images in a directory
def get_all_features(image_dir):
    features_list = []
    image_paths = []
    
    if not os.path.exists(image_dir):
        st.error(f"Directory not found: {image_dir}")
        return [], []
    
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            features = extract_features(img_path)
            features_list.append(features)
            image_paths.append(img_path)
    return features_list, image_paths

# Function to find similar images
def get_similar_images(query_img_path, features_list, image_paths, top_n=5):
    query_features = extract_features(query_img_path)
    similarities = [cosine_similarity(query_features.reshape(1, -1), feat.reshape(1, -1))[0][0] for feat in features_list]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [image_paths[i] for i in top_indices]

# Streamlit Interface
st.title("Image Similarity Finder using VGG16")
st.write("Upload an image to find similar images.")

# Ensure path to images is correct (adjust if needed)
image_dir = os.path.abspath("/content/Images")  # Modify path if necessary
st.write(f"Looking for images in: {image_dir}")

# Check if the directory exists
if os.path.exists(image_dir):
    st.success(f"Directory found: {image_dir}")
else:
    st.error(f"Directory not found: {image_dir}")

# Upload query image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded file temporarily
    temp_path = "temp_query_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load features and image paths
    features_list, image_paths = get_all_features(image_dir)

    if features_list:
        # Find similar images
        similar_images = get_similar_images(temp_path, features_list, image_paths)

        # Display similar images
        st.write("Top similar images:")
        for img_path in similar_images:
            img = Image.open(img_path)
            st.image(img, caption=img_path, use_column_width=True)
    else:
        st.warning("No images found or features could not be extracted.")
