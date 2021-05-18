#!/usr/bin/env python

"""
Image Search using Color Histograms: From a collection of images, extract color histograms and compare to target histogram, to find similar images. 

For directory containing images and specified target image:
  - Get all filepaths of images
  - Get color histogram of target image 
  - For all remaining images:
      - Get color histogram of image
      - Comapare color histogram to target color histogram using chi-square distance
  - Save all filenames and distance to target in a dataframe
  - Save a plot of the target image and the 3 most similar images

Input:
  - -d, --directory, str <directory-of-images> (optional, default: ../data/flowers)
  - -t, --target_img, str <name-of-target-img> (optional, default: image_0001.jpg)

Output stored in ../out:
  - {target_img}_hist.csv: csv file with filename and chi-square distance from target image
  - {target_img}_hist_top3.png: image with target image and top3 closest images 
"""


# LIBRARIES ------------------------------------

# Basics
import os
import sys
import argparse
from tqdm import tqdm

# Utils
sys.path.append(os.path.join(".."))
from utils.img_search_utils import get_paths, plot_similar
           
# Images and data
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# HELPER FUNCTIONS ------------------------------------

def get_histogram(img_path):
    """
    Generate normalised color histogram for RGB image stored in img_path
    Input: 
      - img_path: path to image as str
    Returns:
      - normalised color histogram
    """
    # Load image
    img = cv2.imread(str(img_path))
                   
    # Calculate histogram for RGB color channels 
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
    
    # Normalise the histogram with min max regularisation
    hist_normalised = cv2.normalize(hist, hist, 0,255, cv2.NORM_MINMAX)
    
    return hist_normalised
    

# MAIN FUNCTION ------------------------------------

def main():
    
    # --- ARGUMENT PARSER  AND FILEPATHS ---
    
    # Initialise arguent parser
    ap = argparse.ArgumentParser()
    
    # Input options for path to images
    ap.add_argument("-d", "--directory", help = "Path to directory of images", 
                    required = False, default = "../data/flowers")
    
    # Input option for target image name
    ap.add_argument("-t", "--target_img", help = "Filename of the target image", 
                    required = False, default = "image_0001.jpg")

    # Extract inputs
    args = vars(ap.parse_args())
    img_dir = args["directory"]
    target_img = args["target_img"]
    
    # --- IMAGE SEARCH ---
    
    # Print message
    print(f"\n[INFO] Initialising image search for {target_img} using color histograms.")
    
    # Get filepath to target image
    target_path = os.path.join(img_dir, target_img)
    # Get file paths to all images in directory 
    img_paths = get_paths(img_dir)
    # Remove target path from all image paths
    img_paths.remove(target_path)
    
    # Create empty target data frame for distances
    distances_df = pd.DataFrame(columns=["filename", "chisquare_distance"])
    
    # Get histogram of target image
    target_hist = get_histogram(target_path)

    # Get histogram for all other images and compare to target histogram 
    for img_path in tqdm(img_paths):
        # Get the name of the image
        img_name = os.path.split(img_path)[1]
        # Get the histogram of the image
        img_hist = get_histogram(img_path)
        # Calculate the distance of the image by comparing the target and image histogram
        distance = round(cv2.compareHist(target_hist, img_hist, cv2.HISTCMP_CHISQR), 2)
        # Append filename and distance to dataframe
        distances_df = distances_df.append({"filename": img_name, 
                                            "chisquare_distance": distance}, ignore_index = True)

        
    # Sort data frame by distance and reset index
    distances_df = distances_df.sort_values("chisquare_distance").reset_index()
    
    # --- OUTPUT ---
    
    # Prepare output directory 
    out_dir = os.path.join("..", "out")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # Save data frame in output directory, using target image for filename
    out_df = os.path.join(out_dir, f"{os.path.splitext(target_img)[0]}_hist.csv")
    distances_df.to_csv(out_df)
    
    # Save plot with similar images in output directory, using target image for filename
    out_plot = os.path.join(out_dir, f"{os.path.splitext(target_img)[0]}_hist_top3.png")
    plot_similar(img_dir, target_img, distances_df, out_plot)

    # Print message, and print closest image to target image
    print(f"\n[INFO] Output is saved in {out_dir}, the closest image to {os.path.splitext(target_img)[0]} is:")
    print(distances_df.iloc[0])
    
    
if __name__=="__main__":
    main()