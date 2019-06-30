# Multimodal Embeddings
This repository contains code that creates multimodal embeddings generated from both image and word embeddings. The code follows the methods described in the paper

> [Imagined Visual Representations as Multimodal Embeddings](https://www.researchgate.net/publication/315297247_Imagined_Visual_Representations_as_Multimodal_Embeddings) 

# Dependencies

* Python 3.6
* FastAI v1
* Pytorch v1

## Preprocessing
In this project, we use the [MMID Dataset](http://multilingual-images.org/) as our image source. To extract image embeddings, we first create a pandas datafram mapping image paths to words. To build the dataframes from MMID data, we can run: 
python image_feature_extraction.py --build_df --dict scale-english-01-package scale-english-02-package 

## Extracting Image Embeddings 
To train model and extract features
1. python image_feature_extraction.py --train scale-english-01-package scale-english-02-package
If inference is done in a distributed fashion, this command will merge the features into a single file, create a magnitude file, and enumerate the images, so that they are unique vectors in mangnitude by appending a number label( i.e "_01"). Note, this is only to distinguish image embeddings in Magnitude, and training the multimodal mapping in the following steps will not append this label.  
2. python image_feature_extraction.py --config config.ini --mode magnitude --fea_filepath /nlp/data/dkeren/img_embeddings_rn50-dicts2.txt

To evaluate the image embeddings we perform AverageMax between a pair of word's image embeddings. 

3. python image_features_extraction.py --eval 

## Creating Training Data

## Learning Mapping Regression 

## Evaluation

