#!/usr/bin/env python

"""
Image Search using Image Embeddings: From a collection of images, find k nearest neighbours for a target image using pretrained convolutional neural network (VGG16)

For directory containing images and specified target image:
  - Get all filepaths of images
  - Load pretrained CNN (VGG16)
  - For each image: 
       - Resize it, prepreprocess it
       - Extract its features (embeddings) using the CNN
  - Append features of all images to a feature list
  - Find k nearest neighbors
  - For target image: 
       - Print and save the k nearest neighbors and distances
       - Save a print a plot of the target image and the 3 nearest neighbors

Input: 
  - -d, --directory: str, <path-to-image-directory> (optional, default: ../data/flowers
  - -t --target_image: str, <name-of-target> (optional, default: image_0001.jpg)
  - -k --k_neighbors: int, <number-of-neighbors> (optional, default: 20)

Output saved in ../out:
  - csv file of cosine distances of k nearest neighbors, 1st nearest neighbor printed to command lines
  - Image file of target image and 3 nearest neighbors
"""


# LIBRARIES ------------------------------------

# Basics
import os, sys
import argparse
from tqdm import tqdm

# Utils
sys.path.append(os.path.join(".."))
from utils.img_search_utils import get_paths, plot_similar

# Data analysis
import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Tensorflow
# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Matplotlib for Visualisations
import matplotlib.pyplot as plt 


# HELPER FUNCTIONS ------------------------------------ 
           
def extract_features(img_path, input_shape, model):
    """
    Load an image, preprocess for VGG16, predict features,
    flatten and normalise feautures 
    Input:
      - img_path: file path to an image
      - input_shape: img shape to fit to pretrained model
    Returns:
      - normalized_features: list of normalised features for img
    """
    # Load the image fitted to the input shape
    img = load_img(img_path, target_size=(input_shape[0],input_shape[1]))    
    # Turn image to array
    img_array = img_to_array(img)
    # Add dimension to the beginning of array
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # Preprocess image for VGG16
    preprocessed_img = preprocess_input(expanded_img_array)

    # Extract, flatten and normalise features of the image using the loaded model
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(features)
    
    return normalized_features
    
    
def get_target_neighbors(img_paths, target_index, feature_list, k_neighbors):
    """
    Get the neighbors and distances of the target image
    Input:
      - img_paths: paths to all images
      - target_index: index of target in all images
      - feature_list: list of extracted features for images
      - k_neighbors: number of k neighbors to get for target
      - k_neighbors = number of neighbors to extract
    Returns: 
      - df with filename of nearest neighbors and cosine distance to target
    """
    # Initialise nearest neighbors algorithm 
    neighbors = NearestNeighbors(n_neighbors=k_neighbors, 
                                 algorithm='brute',
                                 metric='cosine').fit(feature_list)

    # From neighbors, extract the neighbors and distances of the target image
    distances, indices = neighbors.kneighbors([feature_list[target_index]])

    # Create empty dataframe to store values
    df = pd.DataFrame(columns=["filename", "cosine_distance"])
    
    # For all indicies of neighbors, start at 1, because 0 is target
    for i in range(1, len(indices[0])):
        # Get the index of the image
        img_index = indices[0][i]
        # Get the name of the image using the index
        img_name = os.path.split(img_paths[img_index])[1]
        # Get thte distance of the image to the target
        distance = distances[0][i]
        # Append filename and distance to dataframe
        df = df.append({"filename": img_name, 
                        "cosine_distance": distance}, ignore_index = True)

    return df


# MAIN FUNCTION ------------------------------------

def main():
    
    # --- ARGUMENT PARSER ---
    
    # Initialise argument parser for output filename
    ap = argparse.ArgumentParser()
    
    # Input options for path, target image, and number of neighbors
    ap.add_argument("-d", "--directory", help = "Path to directory of images", 
                    required = False, default = "../data/flowers")
    ap.add_argument("-t", "--target_img", help = "Filename of the target image", 
                    required = False, default = "image_0001.jpg")
    ap.add_argument("-k", "--k_neighbors", help = "Number of neighbors to extract for target", 
                    required = False, default = 20)
    
    # Extract input parameters
    args = vars(ap.parse_args())
    img_dir = args["directory"]
    target_img = args["target_img"]
    k_neighbors = args["k_neighbors"]
    
    # --- IMAGE SEARCH: FEATURE EXTRACTION AND KNN ---
    
    # Print message
    print(f"\n[INFO] Initialising image search for {target_img} using features extracted from VGG16.")
    
    # Get all file paths and file path and index to target image
    img_paths = get_paths(img_dir)
    target_path = os.path.join(img_dir, target_img)
    target_index = img_paths.index(target_path)
    
    # Define input shape and load pretrained model (VGG16)
    input_shape = (224,244,3)
    model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
    
    # Extract features of all images using VGG16
    feature_list = []
    for img_path in tqdm(img_paths):
        img_features = extract_features(img_path, input_shape, model)
        feature_list.append(img_features)
    
    # Get k nearest neighbors of target image, and store name and distance in df
    distances_df = get_target_neighbors(img_paths, target_index, feature_list, k_neighbors)
    
    # --- OUTPUT ---
    
    # Define output directory
    out_dir = os.path.join("..", "out")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    # Save data frame in output directory
    out_df = os.path.join(out_dir, f"{os.path.splitext(target_img)[0]}_embeddings.csv")
    distances_df.to_csv(out_df) 
    
    # Plot and save target neighbors in output directory
    out_plot = os.path.join(out_dir, f"{os.path.splitext(target_img)[0]}_embeddings_top3.png")
    plot_similar(img_dir, target_img, distances_df, out_plot)
    
    # Print message
    print(f"\n[INFO] Output is saved in {out_dir}, the closest image to {os.path.splitext(target_img)[0]} is:")
    print(distances_df.iloc[0])

    
if __name__=="__main__":
    main()