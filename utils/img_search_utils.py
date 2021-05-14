#!/usr/bin/python

"""
Utility functions for image search
  - get_paths: function to get all images with different extensions in a directory 
  - plot_similar: plot target image and three most similar images (based on distances and filenames stored in df)
"""

# LIBRARIES ------------------------------------

import os
import matplotlib.pyplot as plt


# UTILITY FUNCTIONS ------------------------------------

def get_paths(img_dir):
    """
    Getting filepaths to all images with defined extensions.
    - directory: str, directory where images are stored  
    Returns:  list of filepaths as str to images in directory
    """
    # Define list of valid extensions
    extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    # Create empty list
    paths = []
    # Initliase counter
    counter = 1
    
    # Create list of filepaths
    for img_dir, directories, filenames in os.walk(img_dir):
        for filename in filenames:
            # Keep only those with valid extensions
            if any(ext in filename for ext in extensions):
                paths.append(os.path.join(img_dir, filename))
                # Increment counter
                counter += 1
                
    paths = sorted(paths)
    
    return paths


def plot_similar(img_dir, target_img, df, out_plot):
    """
    Create and Save plot of target image and the three most similar images 
    - img_dir: directory to all images
    - target_img: name of target image
    - df: dataframe containing filenames and distances 
    - out_plot: path where plot should be stored
    Saves: plot in out_plot
    """
    
    # Get path to the target image
    target_path = os.path.join(img_dir, target_img)
    
    # Get the path to the three closest images
    top1, top2, top3 = df["filename"][0:3]
    path1 = os.path.join(img_dir, top1)
    path2 = os.path.join(img_dir, top2)
    path3 = os.path.join(img_dir, top3)

    # Define grid
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    ax4 = plt.subplot2grid((2, 3), (1, 2))

    # Plot target image on top
    ax1.set_title(f'Target Image: {target_img}')
    ax1.imshow(plt.imread(target_path))
    
    # Plot the three clostest images
    ax2.set_title(f"1st similar image")
    ax2.imshow(plt.imread(path1))
    ax3.set_title(f"2nd similar image")
    ax3.imshow(plt.imread(path2))
    ax4.set_title(f"3rd similar image")
    ax4.imshow(plt.imread(path3))

    # Save figure
    plt.tight_layout(pad=2.0)
    plt.savefig(out_plot)
    
    
if __name__=="__main__":
    pass