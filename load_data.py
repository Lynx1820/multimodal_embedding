"""
Purpose:
 - Collect image embeddings in one txt file, convert to Magnitude in command line 
 - Create the training set (x_train, y_train)
"""
import sys
import numpy as np
import pickle
import pandas as pd 
import os
from pymagnitude import *

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

def create_image_embedding_resnet(data_path, folder_name):
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
        for 
        print("Done loading from pandas")
        #averaged = np.array([])
        start = 0
        for i in range(words.shape[0]-1):
            # only process English words, which start with 'row'
            if words[i] != words[i+1]:
                end = i+1
                img_embedding = embeddings[start:end]
                # average pooling to create one single image embedding
                average_embedding = img_embedding.sum(axis=0) / img_embedding.shape[0]
                if words[i] in word_to_index: 
                    #print("found repeated word" + words[i])
                    averaged[word_to_index[words[i]],:] = (averaged[word_to_index[words[i]],:] + average_embedding) / 2
                    continue
                #average_embedding = np.insert(average_embedding, 0, words[i][0])
                word_to_index[words[i]] = av_index
                av_index += 1
                if init == 1:
                    averaged = average_embedding
                    init = 0 
                averaged = np.vstack((averaged,average_embedding))
                start = i+1
                # save all embeddings to txt, convert txt to magnitude in cmd line 
    with open(data_path, 'a') as dfile:
        for word in word_to_index: 
            dfile.write(word + "\t")
            np.savetxt(dfile, averaged[word_to_index[word],:].reshape(1,averaged[word_to_index[word],:].shape[0]), fmt="%s")
    #print(word_to_index.keys())
                
    print("Done average pooling, words" )
    print(len(word_to_index.keys())) 
# 
def create_train_set(word_magnitude_file,image_magnitude_file):
    """
    create the train set (x_train, y_train)
    @return x_train, y_train
    """
    words = pd.read_csv('/nlp/data/dkeren/resnet34_img_embedding.txt', sep=' ', header=None).values
    # save all words in a txt file k
    np.savetxt('/nlp/data/dkeren/words.txt', words[:,0], fmt="%s")
    word_dict = Magnitude(word_magnitude_file)
    #img_dict = Magnitude('/data1/embeddings/pymagnitude/image.magnitude')
    img_dict = Magnitude(image_magnitude_file)
    # TODO: skip over words with all NaNs    
 
    # create a file of processed words (no annotations of translation)
    # query for processed words' embeddings
    for i in range(words.shape[0]):
        unprocessed_word = words[i][0]
        # convert word, e.g row-writings to writings 
        if "row" in words[i][0]:
            phrase = words[i][0].split('-')[1]
        if "_" in words[i][0]:
            word_list = phrase.split('_')
            word = ""
            for i in range(len(word_list)):
                word += word_list[i]
                if i < len(word_list)-1:
                    word += " "
            phrase = word 
        
        with open('words_processed.txt', 'a') as f:
            f.write("{}\n".format(phrase))
        word_embedding = word_dict.query(phrase) #query word embedding for image word
        check_nan = np.isnan(word_embedding)
        all_nan = check_nan[check_nan==True].shape[0] #number of nans
        if all_nan == word_embedding.shape[0]: print("Nan: " + phrase)
        img_embedding = img_dict.query(unprocessed_word)
        check_nan = np.isnan(img_embedding)
        all_nan = check_nan[check_nan==True].shape[0]
        # check if a word has valid image vectors 
        # valid: image vectors doesn't contain all NaNs
        # check_nan = np.isnan(img_embedding)
        # all_nan = check_nan[check_nan==True].shape[0]
        # if all_nan == img_embedding.shape[0]:
            
        # add to x_train and y_train
        with open('/nlp/data/dkeren/x_train1.txt', 'a') as f:
            np.savetxt(f, word_embedding.reshape(1, word_embedding.shape[0]))
        with open('/nlp/data/dkeren/y_train1.txt', 'a') as f:
            np.savetxt(f, img_embedding.reshape(1, img_embedding.shape[0]))

#create_image_embedding(folder_name)
#folder with image vectors: '/data1/minh/data'
#current folder with sample fasttext: '~/data/fasttext_sample.magnitude'
#folder with image magnitude: '/data1/mihn/magnitude/image.magnitude'
#folder_path = "/nlp/data/dkeren/data_chunks"

folder_path = "/nlp/data/dkeren/" + sys.argv[1]
data_path = '/nlp/data/dkeren/img_embedding_' + sys.argv[1] + ".txt"
# TODO: for later
word_magnitude_file = '/nlp/data/dkeren/crawl-300d-2M.magnitude'
image_magnitude_file = '/nlp/data/dkeren/img.magnitude'
if sys.argv[1] == 'train': 
    create_train_set(word_magnitude_file,image_magnitude_file)
else: 
    create_image_embedding_resnet(data_path, folder_path)
