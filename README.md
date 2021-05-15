# Image Search: Colour Histograms vs Feature Embeddings

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion)

## Description

> This project relates to Assignment 2: Visual Image Search of the course Visual Analytics

The purpose of the project was to develop a tool to find similar images to a given target image. Such a tool could be useful to identify duplicates or sort images. Further, it may be practically implemented as an image recommender, i.e., *if you like this image, you might also like these images*. 

Two scripts were developed, which use different methods to find similar images for a target image: (1) using colour histograms and (2) using transfer learning, i.e., feature embeddings extracted from the pre-trained model VGG16. The aim of developing scripts for two different methods was also to compare their outputs and usability, which are discussed below.


## Methods

### Data
For this project, the [Oxford-17 flowers image dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) was used. This dataset contains 80 images for 17 different categories (a total of 1360 images). 

### 1.  Image search using colour histograms
Images can be represented by their distribution of color. In RGB colour space, colour histograms display how often each pixel intensity (0-255) occurs in each of the three colour channels for a given image. Thus, colour histograms were extracted for each of the images, and compared to the histogram of the target image. Similarity was measured using the chi-square distance measure. 

### 2. Image search using image embeddings
Pre-trained CNNs can be used to extract image embeddings. In other words, pre-trained weights are used to predict a dense feature space, i.e., vector for each image. These vectors can be compared to find images that are visually similar. In this project, the pre-trained model VGG16 is used to extract features of each image. These feature representations are then fed into a k-Nearest-Neighbour algorithm, to find the k most similar images to a given target image.  


## Repository Structure

```
|-- data/                               # Directory for input data
    |-- flowers.zip                     # .zip data to run scripts 

|-- out/                                # Directory for output from scripts
    |-- image_0001_embeddings.csv       # Example output: csv file with similar images using feature embeddings
    |-- image_0001_embeddings_top3.png  # Example output: png with target image and 3 clostest images using feature embedddings
    |-- image_0001_hist.csv             # Example output: csv file with silimiar images using color histograms
    |-- image_0001_hist_top3.png        # Exmaple output: png with target image and 3 clostest images using histograms

|-- src/                                # Image search scripts
    |-- img_search_embeddings.py        # Image search using feature embeddings
    |-- img_search_histogram.py         # Image search using color histograms

|-- utils/                              # Utilities
    |-- img_search_utils.py             # Utility script, with functions used across scripts

|-- README.md
|-- create_venv.sh                      # Bash script to recreate virtual environment 
|-- requirements.txt                    # Dependencies, installed in virtual environment
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
bash create_venv.sh

# activate virtual environment 
source venv_imagesearch/bin/activate
```


### 2. Data
A zip file of the [Oxford-17 flowers image dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) called `flowers.zip` is in the `data` directory, and should be unzipped to run the script. The script can also be run on a different set of images with the following extensions .jpg, .JPG, .jpeg, .JPEG, .png, .PNG,  by setting the `--path` argument (see below). 

```bash
# unzip files
cd data
unzip flowers.zip
cd ..
```

### 3.1. Image search using color histograms: img_search_histogram.py
For image search using colour histograms, the script `img_search_histogram.py` should be run from the `src/` directory:

```bash
# move into src directory before running the script
cd src/

# running script on default parameters
python3 img_search_histogram.py

# running script with specified parameters
python3 img_search_histogram.py -t image_0002.jpg
```

__Parameters:__
- `-d, --directory`: *str, optional, default:*  `../data/flowers`\
  Path to directory where images are stored. Note that running default requires unzipping flowers.zip files (see above). 

- `-t, --target_img`: *str, optional, default:* `image_0001.jpg`\
  Target image, for which all other images should be compared to find the most similar ones. 


__Output__ saved in `out/`:
- `{target_img}_hist.csv`\
  CSV file with filenames and chi-square distances of all images to the target image

- `{target_img}_hist_top3.png`\
  Image with target image and 3 most similar images. 


### 3.2. Image search using image embeddings: img_search_embeddigs.py
For image search using feature embeddings, the script `img_search_embeddings.py` should be run from the `src/` directory:

```bash
# move into src directory before running the script
cd src/

# running script on default parameters
python3 img_search_embeddings.py

# running script with specified parameters
python3 img_search_embeddings.py -t image_0002.jpg
```

__Parameters:__
- `-d, --directory`: *str, optional, default:*  `../data/flowers`\
  Path to directory where images are stored. Note that running default requires unzipping flowers.zip files (see above). 

- `-t, --target_img`: *str, optional, default:* `image_0001.jpg`\
  Target image, for which all other images should be compared to find the most similar ones. 

- `-k, --k_neighbors`: *int, optional, default:* `20`\
  Number of k neighbors to extract and save distances for. 


__Output__ saved in `out/`:
- `{target_img}_embeddings.csv`\
  CSV file with filenames and cosine distances of k nearest images to the target image. 

- `{target_img}_embeddings_top3.png`\
  Image with target image and 3 most similar images.


## Results and Discussion 
Both scripts were used to find similar images to two different target images (displayed below). From these examples, it can be seen that image search using colour histograms could identify images that are fairly similar in their colour. However, the use of feature embeddings actually allowed to find images of the same kind of flower. 

Colour histograms cannot take into account shapes, textures or any spatial relations. However, feature embeddings can take into account more complex aspects of images, such as shapes, textures and spatial relations. Thus, colour histograms may be useful to identify identical images, as they will have the same colour distribution, but feature embeddings are more useful to find rather semantically similar images.   

__Three most similar images to image_0001.jpg:__

 (1) using colour histograms | (2) using feature embeddings
:-------------------------:|:-------------------------:
![](https://github.com/nicole-dwenger/cdsvisual-imagesearch/blob/master/out/image_0001_hist_top3.png)  |  ![](https://github.com/nicole-dwenger/cdsvisual-imagesearch/blob/master/out/image_0001_embeddings_top3.png)


__Three most similar images to image_0300.jpg:__

 (1) using colour histograms | (2) using feature embeddings
:-------------------------:|:-------------------------:
![](https://github.com/nicole-dwenger/cdsvisual-imagesearch/blob/master/out/image_1320_hist_top3.png)  |  ![](https://github.com/nicole-dwenger/cdsvisual-imagesearch/blob/master/out/image_1320_embeddings_top3.png)

