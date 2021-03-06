from fastai.vision import *
from fastai.metrics import error_rate
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils import data
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from process_eval_set import get_eval_set_dict 
from evaluation import compute_pair_sim
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import configparser
from pymagnitude import *
import argparse
import subprocess
import time

def build_dataframe(dict_fn, paths, filter_mode = True):
    img_paths = []
    trans = []
    word_magnitude = None
    if filter_mode: 
        word_magnitude =  Magnitude(paths['word_magnitude']) 
        
    with open(paths['mmid_dir'] + "/" + dict_fn + "/index.tsv") as f:
        lines = f.read().splitlines()
        for line in lines:
            line = line.replace("\\t", "\t")
            english_translation, index = line.split('\t')
            if filter_mode and english_translation not in word_magnitude: continue
            ends = ["01","02","03","04","05","06","07","08","09","10"]
            for img_num in ends:
                path = dict_fn + "/" + str(index) + "/" + img_num + ".jpg"
                exists = os.path.isfile(paths['mmid_dir'] + "/" + path)
                # check to see whether the path exist
                if not exists: 
                    continue
                img_paths.append(path) 
                trans.append(english_translation)

    #dataframe with paths and translations
    df = pd.DataFrame({'paths': img_paths, 'trans' : trans}).dropna()
    df.to_csv(dict_fn + "-train_df.csv", sep='\t')

class SaveFeatures():
    features = None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        self.hook.remove()
def train(train_df): 

    # default transformations applied below 
    my_data = ImageDataBunch.from_df(paths['code_dir'], train_df, ds_tfms=get_transforms(), size=224, bs=params.bs, folder=paths['mmid_dir']).normalize(imagenet_stats)
    learn = cnn_learner(my_data, models.resnet50, metrics=error_rate)

    #This changes the forwards layers of the model 
    learn.fit_one_cycle(params.train_epochs,params.lr)
        #saving model 
    modelname = paths['data_dir'] + "/" + params.model_name
    print("Saving model to: " + modelname)
    learn.save(modelname)
    return learn, my_data

def extract_features(learn, my_data):
    trans = []
    features = []
    iter = 1000
    for i in range(len(my_data.train_ds)):
        if i % iter == 0:
            print("Working on iteration:"  + str(i))
        sf = SaveFeatures(learn.model[1][5]) 
        _= learn.predict(my_data.train_ds[i][0])
        trans.append(str(my_data.train_ds[i][1]))
        features.append(sf.features[0])
    return trans, np.array(features)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Image Feature Extraction')
    parser.add_argument("--config", type=str, default=None, help= 'config file to specify paths')
    parser.add_argument("--dict", type=str, nargs='+', default=None, help='List of image disctionaries to train on')
    parser.add_argument("--fea_filepath", type=str, default=None, help='Features\' filename')
    parser.add_argument("--workers", type=int, default=1, help="number of processes to do work in the distributed cluster")
    parser.add_argument("--pid", type=int, default=None, help="set to true when just extracting features" )
    parser.add_argument("--mode",choices=['train', 'eval', 'partition', 'magnitude', 'build_df', 'inference'], help="build_df: preprocess data and create embeddings, eval: evaluate embeddings")
    parser.add_argument("--train_epochs", type=int, default=8, help="number of epochs to train before extracting features") 
    parser.add_argument("--model_name", type=str, default="model_rn50", help="model name") 
    parser.add_argument("--split_df", type=str, default=None, help="file to inference on")
    parser.add_argument("--full_df", type=str, default=None, help="df model was originally trained on")
    #Model Paramaters
    parser.add_argument("--bs", type=int, default=64, help="batch size for training") 
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate") 
    params = parser.parse_args()
    print("Parsing Arguments")
    # check params 
    assert os.path.isfile(params.config)
    config = configparser.ConfigParser()
    config.read(params.config)
    paths = config['PATHS']
    ## TODO: assert for all partials workers
    assert os.path.isfile(paths['word_magnitude'])
    if params.mode == 'train':
        for dict_fn in params.dict:
            assert os.path.isdir(paths['mmid_dir'] + "/" + dict_fn) 
        # TODO: if everythings works then maybe don't save the file or delete
        df_list = []
        for dict_fn in params.dict:
            df_list.append(pd.read_csv(dict_fn + '-train_df.csv', sep='\t', index_col=[0]))
        all_dfs = pd.concat(df_list).reset_index().drop(columns=['index'])
        all_dfs.to_csv("all-train_df.csv", sep='\t')
        translations, features = extract_features(train(all_dfs))
        filename = paths['data_dir'] + "/img_embeddings_rn50-dicts" + str(len(params.dict)) + ".txt"
        print("Saving features to file: " + filename)
        with open(filename, 'a') as f:
            for word, arr in zip(translations,features):
                f.write(word + "\t")
                np.savetxt(f, arr.reshape(1,len(arr)), delimiter=' ')
        exit(0)
    elif params.mode == 'inference':
        print("Using modelname: " + params.model_name)
        print("Inference on the following file" + params.split_df)
        train_df = pd.read_csv(params.split_df, sep='\t', index_col=[0])
        train_df = np.array_split(train_df, 5)[params.pid].reset_index().drop(columns=['index'])
        #print(train_df.head())
        # split_data = ImageList.from_df(train_df,paths['code_dir'], folder=paths['mmid_dir'])
        split_data = ImageDataBunch.from_df(paths['code_dir'], train_df, size=224, bs=params.bs, folder=paths['mmid_dir']).normalize(imagenet_stats)
        learn = load_learner(paths['code_dir'])
        translations, features = extract_features(learn, split_data) 
        filename = paths['data_dir'] + "/img_embeddings-" + params.split_df[:-4] + "-" + str(params.pid) + ".txt"
        print("Saving features to file: " + filename)
        with open(filename, 'a') as f:
            for word, arr in zip(translations,features):
                f.write(word + "\t")
                np.savetxt(f, arr.reshape(1,len(arr)), delimiter=' ')
        print("Done saving")
    elif params.mode == 'build_df': 
        df_list = []
        for dict_fn in params.dict: 
            build_dataframe(dict_fn, paths)
    elif params.mode == 'magnitude': 
        fn = paths['data_dir'] + "/full_embeddings_rn50_numbered.txt"
        magnitude_fn = paths['data_dir'] + "/full_img_embeddings_rn50_numbered.magnitude"
        full_file = open(fn, 'w')
        words = Counter()
        print("Writing to %s", fn)
        with open(params.fea_filepath, 'r') as file: 
            for line in file:
                word, emb = line.split('\t')
                if words[word] == 0: 
                    full_file.write(line)
                    words[word] += 1
                else: 
                    new_line = word + "_" + str(words[word]) + '\t' + emb
                    full_file.write(new_line)
                    words[word] += 1
        full_file.close()
        print("done writing new dictionaries")
        cmd = ("python -m pymagnitude.converter -i " + fn +  " -o "  + magnitude_fn).split()

        try: 
            print("running: " + str(cmd))
            subprocess.check_output(cmd)       
        except: 
            raise Exception("There was an error running" + str(cmd))
    elif params.mode == 'eval':
        magnitude_fn = paths['data_dir'] + "/full_img_embeddings_rn50_numbered.magnitude"
        eval_set_dict = get_eval_set_dict(paths)
        embeddings = Magnitude(magnitude_fn)
        for eval_name, eval_set in eval_set_dict.items():
            words_sim = []
            human_ratings = []
            for i in range(eval_set.shape[0]):
                word1 = eval_set[i][0]
                word2 = eval_set[i][1]
                rating = eval_set[i][2]
                if word1 in embeddings and word2 in embeddings:
                    emb1 = [embeddings.query(word1) ]
                    emb2 = [embeddings.query(word2) ]
                    for label in range(1,10):
                        tmp1 = word1 + "_" + str(label) 
                        tmp2 = word2 + "_" + str(label)
                        if tmp1 in embeddings: 
                            emb1.append(embeddings.query(tmp1))
                        if tmp2 in embeddings: 
                            emb2.append(embeddings.query(tmp2))
                    cos_sim = sum(np.amax(cosine_similarity(np.array(emb1), np.array(emb2)), axis=1))/len(emb1)
                    words_sim.append(cos_sim)
                    human_ratings.append(rating)
                    print("words: " + word1 + " " +word2 + " " + str(rating) + " " + str(cos_sim))
            print(words_sim)
            if len(words_sim) == 0:
                continue
            cor, pval = stats.spearmanr(words_sim, human_ratings)
            print("Correlation for {}: {:.3f}, P-value: {:.3f}".format(eval_name, cor, pval))

    else:  
        print("options are: eval, build, partition, merge")
        exit(0)


