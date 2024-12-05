import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

# Load the VGG16 model without the top classification layer
@st.cache_resource
def load_model():
    return VGG16(weights='imagenet', include_top=False)

model = load_model()

# Extract features for all images in the dataset
@st.cache_data
def load_features(image_dir):
    features_list = []
    image_paths = []
    
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        image_paths.append(img_path)
        
        # Load and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features using VGG16
        features = model.predict(img_array)
        features_list.append(features.flatten())
        
    return features_list, image_paths

# Function to find similar images
def get_similar_images(query_image_path, features_list, image_paths, top_n=5):
    # Load and preprocess the query image
    query_img = load_img(query_image_path, target_size=(224, 224))
    query_img_array = img_to_array(query_img)
    query_img_array = np.expand_dims(query_img_array, axis=0)
    query_img_array = preprocess_input(query_img_array)
    
    # Extract features from the query image
    query_features = model.predict(query_img_array).flatten().reshape(1, -1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_features, features_list)
    similar_indices = np.argsort(similarities[0])[::-1][:top_n]
    
    return [image_paths[i] for i in similar_indices]

# Main Streamlit app
def main():
    st.title("Image Similarity Finder using VGG16")
    
    # Directory containing the dataset images
    image_dir = "path/to/your/image/dataset"
    
    # Load the features and image paths
    features_list, image_paths = load_features(image_dir)
    
    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        query_image_path = f"temp_{uploaded_file.name}"
        with open(query_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(query_image_path, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Find Similar Images"):
            similar_images = get_similar_images(query_image_path, features_list, image_paths)
            
            st.write("Top similar images:")
            for img_path in similar_images:
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
        
        # Cleanup temporary file
        os.remove(query_image_path)

if __name__ == "__main__":
    main()
