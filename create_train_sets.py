"""
1;5202;0cPurpose:
 - Collect image embeddings in one txt file, convert to Magnitude in command line 
 - Create the training set (x_train, y_train)
"""
import sys
import numpy as np
import pickle
import pandas as pd 
import os
from pymagnitude import *
import configparser
from collections import Counter


def create_image_embedding_resnet(data_path, folder_name, no_mean = False):
    """
    create one image embedding for each word by average pooling all image feature vectors
    @save img_embedding: a numpy array of image embeddings 
    """
    # read the files that contain dfs with words and image embeddings - all distributed in chunchs
    # TODO: handle duplicates or not (maybe Magnitude will eventually handle this? 
    folders = os.listdir(folder_name)
    all_words = {}
    word_to_index = {}
    init = 1
    av_index = 0
    for f in folders:
        print("Folder name: {}".format(f))
        split = pd.read_csv(folder_name + "/" + f, sep='\t', header=None).values
        
        words = split[:, 0]
        temp = split[:, 1]
        embeddings = np.asarray([np.asarray(x.split(' ')) for x in temp]).astype(float)
        
        print("Done loading from pandas")

        start = 0
        dfile =  open(data_path, 'a')
        for i in range(words.shape[0]-1):
            # only process English words, which start with 'row'
            if no_mean: 
                dfile.write(words[i] + "-"+ str(start) + "\t")
                np.savetxt(dfile,embeddings[i].reshape(1,average_embedding.shape[0]), fmt="%s")
            if words[i] != words[i+1]:
                if no_mean: 
                    start = 0
                    continue
                end = i+1
                img_embedding = embeddings[start:end]
                # average pooling to create one single image embedding
                average_embedding = img_embedding.sum(axis=0) / img_embedding.shape[0]

                start = i+1
                dfile.write(words[i] + "\t")
                np.savetxt(dfile, average_embedding.reshape(1,average_embedding.shape[0]), fmt="%s")
                # save all embeddings to txt, convert txt to magnitude in cmd line 
                
    print("Done average pooling, words" )

def create_train_set():
    """
    create the train set (x_train, y_train)
    @return x_train, y_train
    """
    words = pd.read_csv(paths['image_embedding'], sep='\t', header=None).values
    # save all words in a txt file k
    word_dict = Magnitude(paths['word_magnitude'])
    
    # create a file of processed words (no annotations of translation)
    for i in range(words.shape[0]):
        phrase = words[i][0].replace('_', ' ')
        with open(paths['code_dir'] + '/words_processed.txt', 'a') as f:
            f.write("{}\n".format(phrase))
        word_embedding = word_dict.query(phrase) #query word embedding for image word
        img_embedding = words[i][1]
        # TODO: paramatize the max number of images

        # add to x_train and y_train
        with open(paths['x_train'], 'a') as f:
            np.savetxt(f, word_embedding.reshape(1, word_embedding.shape[0]))
        with open(paths['y_train'], 'a') as f:
            f.write("{}\n".format(img_embedding))


#folder with image vectors: '/data1/minh/data'
#folder with image magnitude: '/data1/mihn/magnitude/image.magnitude'

parser = argparse.ArgumentParser(description='Image Feature Extraction')
parser.add_argument("--config", type=str, required=True, default=None, help= 'config file to specify paths')
params = parser.parse_args()
config = configparser.ConfigParser()
config.read(params.config)
paths = config['PATHS']


print("Creating Training Data")
print("Reading embeddings from: " + paths['image_embedding'])
print("Saving to data to: " + paths['x_train'] + " and " + paths['y_train'])
print("Saving processed words to: " + paths['code_dir'] + "/word_processed.txt ")
create_train_set()