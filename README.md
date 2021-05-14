# Image Search 

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | 

## Description

> This project relates to Assignment 2: Visual Image Search of the course Visual Analytics

This repository contains scripts for two different methods to find images which are similar to a target image: (1) using color histograms and (2) using feature embeddings extracted the pretrained CNN VGG16. The aim of developing scripts for two different methods was also to compare of their outputs and usability, which are discussed below. Finding similar images can have simple, practical motivations, e.g. sorting images, but it can also be used for e.g. image recommendation.


## Methods

### Data
For this project, the [Oxford-17 flowers image dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/), containing 80 images for 17 different categories (total of 1360). 

### Image Search using Color Histograms
Images can be represented by their distribution of color. In an RGB color space, color histograms display the how often each pixel intensity (between 0-255) occurs in each of the three color channels for a given image. These distributions can be compared using different distance metrics. The script in this repository extracts color distributions of each of the images and identifies similar images using the chi-square distance measure.

### Image Search using Image Embeddings
Pretrained CNNs can be used to extract image embeddings. In other words, pretrained weights are predict a dense feature space, i.e. vector for a given image. These vectors can be compared to find images which are visually similar. To do this, images were resized to fit to the input shape of VGG16, i.e. 224x224x3. The script in this repository extracts features of each resized image using the pretrained CNNs VGG16 and finds similar images using the k-Nearest-Neighbour algorithm. 


## Repository Structure

```
|-- data/                               # directory for input data
    |-- flowers.zip                     # .zip data to run scripts 

|-- out/                                # directory for output from scripts
    |-- image_0001_embeddings.csv       # example output: csv file with similar images using feature embeddings
    |-- image_0001_embeddings_top3.png  # example output: png with target image and 3 clostest images using feature embedddings
    |-- image_0001_hist.csv             # example output: csv file with silimiar images using color histograms
    |-- image_0001_hist_top3.png        # exmaple output: png with target image and 3 clostest images using histograms

|-- src/                                # image search scripts
    |-- img_search_embeddings.py        # image search using feature embeddings
    |-- img_search_histogram.py         # image search using color histograms

|-- utils/                              # utilities
    |-- img_search_utils.py             # utility script, with functions used across scripts

|-- README.md
|-- create_venv.sh                      # bash script to recreate virtual environment 
|-- requirements.txt                    # dependencies, installed in virtual environment
```


## Usage 

**!** The scripts have only been tested on Linux, using Python 3.6.9. 


### 1. Cloning the Repository and Installing Dependencies

To run the scripts, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create a virtual environment called `venv_imagesearch` with all necessary dependencies, listed in the `requirements.txt` file. The following commands can be used:

```bash
# cloning the repository
git clone https://github.com/nicole-dwenger/cdsvisual-imagesearch.git

# move into directory
cd cdsvisual-imagesearch

# install virtual environment
bash create_vision_venv.sh

# activate virtual environment 
source venv_imagesearch/bin/activate
```


### 2. Data
A zip file of the [Oxford-17 flowers image dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) called `flowers.zip` is in the `data` directory, and should be unzipped to run the script. If you wish to run the script on your own image files, they should have one of the following extensions (.jpg, .JPG, .jpeg, .JPEG, .png, .PNG), and the `--path` argument needs to be specified when running the script (see below). 

```bash
# unzip files
cd data
unzip flowers.zip
cd ..
```

### 3.1. Image Search Using Color Histograms: img_search_histogram.py

The script should be run from the `src/` directory:
```bash
# move into src directory before running the script
cd src/

# running script on default parameters
python3 img_search_histogram.py

# running script with specified parameters
python3 img_search_histogram.py -t image_0002.jpg
```

__Parameters:__
- *-d, --directory : str, optional, default:*  `../data/flowers`\
  Path to directory where images are stored. Note that running default requires unzipping flowers.zip files (see above). 

- *-t, --target_img : optional, default:* `image_0001.jpg`\
  Target image, for which all other images should be compared to find the most similar ones. 


__Output:__

The following output will be saved in a directory called `/out`. Examples can be found in `/out` in this directory.

- *{target_img}_hist.csv:*\
  CSV file with filenames and chi-square distances of all images to the target image

- *{target_img}_hist_top3.png:*\
  Image with target image and 3 most similar images. 


### 3.2. Image Search Using Image Embeddings: img_search_embeddigs.py

The script should be run from the `src/` directory:
```bash
# move into src directory before running the script
cd src/

# running script on default parameters
python3 img_search_embeddings.py

# running script with specified parameters
python3 img_search_embeddings.py -t image_0002.jpg
```

__Parameters:__
- *-d, --directory : str, optional, default:*  `../data/flowers`\
  Path to directory where images are stored. Note that running default requires unzipping flowers.zip files (see above). 

- *-t, --target_img : optional, default:* `image_0001.jpg`\
  Target image, for which all other images should be compared to find the most similar ones. 

- *-k, --k_neighbors : optional, default:* `20`\
  Number of k neighbors to extract and save distances for. 


__Output:__

The following output will be saved in a directory called `/out`. Examples can be found in `/out` in this directory.

- *{target_img}_embeddings.csv:*\
  CSV file with filenames and cosine distances of k nearest images to the target image. 

- *{target_img}_embeddings_top3.png:*\
  Image with target image and 3 most similar images.


## Results and Discussion 

Both scripts have been run two images of the flowers dataset. These images and the three most similar images that have been identified using color histograms and image embeddings are displayed below. From these examples, it can be seen that image search using color histograms can find images of flowers which are fairly similar in e.g. their color. However, only comparison of feature embeddings actually allows finding images of flowers of the same kind.  

Color histrograms can not take into account shapes, textures or any spatial relations. Feature embeddings can take into account more complex aspects of images, such as shapes, textures and spatial relations. Thus, color histograms may be useful to identify identical images, as they will have the same color distribution, but to find rather semantically similar images, feature embeddings might be more useful. 

__Three most similar images to image_0001.jpg:__

Using color histograms:

<img src="https://github.com/nicole-dwenger/cdsvisual-imagesearch/blob/master/out/image_0001_hist_top3.png" alt="hist1" width="450"/>

Using feature embeddings from VGG16:

<img src="https://github.com/nicole-dwenger/cdsvisual-imagesearch/blob/master/out/image_0001_embeddings_top3.png" alt="hist1" width="450"/>

__Three most similar images to image_0300.jpg:__

Using color histograms:

<img src="https://github.com/nicole-dwenger/cdsvisual-imagesearch/blob/master/out/image_1320_hist_top3.png" alt="hist1" width="450"/>

Using feature embeddings from VGG16:

<img src="https://github.com/nicole-dwenger/cdsvisual-imagesearch/blob/master/out/image_1320_embeddings_top3.png" alt="hist1" width="450"/>
