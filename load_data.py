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
def create_image_embedding(folder_name):
    """
    create one image embedding for each word by average pooling all image feature vectors
    @save img_embedding: a numpy array of image embeddings 
    """
    # read the files that contain words and their image embeddings
    # TODO: handle duplicates or not (maybe Magnitude will eventually handle this? 
    folders = os.listdir(folder_name)
    for f in folders:
        print("Folder name: {}".format(f))
        words = pd.read_csv('/data1/minh/data/'+f, sep=' ', header=None).values
        print("Done loading from pandas")
        
        start = 0
        for i in range(words.shape[0]-1):
            # only process English words, which start with 'row'
            if words[i][0] != words[i+1][0]:
                end = i+1
                data_path = '/data1/minh/multimodal/img_embedding.txt'
                img_embedding = words[start:end,1:]
                # average pooling to create one single image embedding
                average_embedding = img_embedding.sum(axis=0) / img_embedding.shape[0]
                average_embedding = np.insert(average_embedding, 0, words[i][0])
                # save all embeddings to txt, convert txt to magnitude in cmd line 
                with open(data_path, 'a') as f:
                    np.savetxt(f, average_embedding.reshape(1, average_embedding.shape[0]), fmt="%s")
                start = i+1
            
            if 'column-' in words[i+1][0]:
                print("Number of English words: {}".format(i/10))
                break
    print("Done average pooling")

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
        
        #df_split = np.load(folder_name + "/" + f, allow_pickle = True)
        #words = df_split['arr_0']
        #features = df_split['arr_1']
        words = split[:, 0]
        temp = split[:, 1]
        embeddings = np.asarray([np.asarray(x.split(' ')) for x in temp]).astype(float)
        
        print("Done loading from pandas")
        #averaged = np.array([])
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
    
    #print(word_to_index.keys())
                
    print("Done average pooling, words" )
    print(len(word_to_index.keys())) 
def create_train_set():
    """
    create the train set (x_train, y_train)
    @return x_train, y_train
    """
    words = pd.read_csv(paths['image_embedding'], sep=' ', header=None).values
    # save all words in a txt file k
    word_dict = Magnitude(paths['word_magnitude'])
    img_dict = Magnitude(paths['image_magnitude'])
    # TODO: skip over words with all NaNs    
    
    # create a file of processed words (no annotations of translation)
    # query for processed words' embeddings
    wrd_counter = Counter()
    for i in range(words.shape[0]):
        phrase = words[i][0].replace('_', ' ')

        # TODO: comment out, just keep to handle old code 
        if "row" in words[i][0]:
             phrase = phrase.split('-')[1]
        # if "_" in words[i][0]:
        #     word_list = phrase.split('_')
        #     word = ""
        #     for i in range(len(word_list)):
        #         word += word_list[i]
        #         if i < len(word_list)-1:
        #             word += " "
        #     phrase = word 
        
        with open(paths['code_dir'] + '/words_processed.txt', 'a') as f:
            f.write("{}\n".format(phrase))
        word_embedding = word_dict.query(phrase) #query word embedding for image word
        # TODO: paramatize the max number of images

        if wrd_counter[phrase] == 0: 
            img_embedding = img_dict.query(phrase)
            wrd_counter[phrase] += 1
        else: 
            phrase_num = phrase + "_" + str(wrd_counter[phrase])
            img_embedding = img_dict.query(phrase_num)
            wrd_counter[phrase] += 1
            
        # add to x_train and y_train
        with open(paths['x_train'], 'a') as f:
            np.savetxt(f, word_embedding.reshape(1, word_embedding.shape[0]))
        with open(paths['y_train'], 'a') as f:
            np.savetxt(f, img_embedding.reshape(1, img_embedding.shape[0]))

#create_image_embedding(folder_name)
#folder with image vectors: '/data1/minh/data'
#current folder with sample fasttent: '~/data/fasttext_sample.magnitude'
#folder with image magnitude: '/data1/mihn/magnitude/image.magnitude'
#folder_path = "/nlp/data/dkeren/data_chunks"

folder_path = "/nlp/data/dkeren/" + sys.argv[1]
data_path = '/nlp/data/dkeren/img_embedding_' + sys.argv[1] + "2.txt"


if len(sys.argv) < 2:
    print("Need config file")
    exit(1)
config = configparser.ConfigParser()
config.read(sys.argv[2])
paths = config['PATHS']

if sys.argv[1] == 'train': 
    create_train_set()
else: 
    if len(sys.argv) > 2 and sys.argv[3] == 'no_mean':
        create_image_embedding_resnet(data_path, folder_path,True)
    else: 
        create_image_embedding_resnet(data_path, folder_path)
