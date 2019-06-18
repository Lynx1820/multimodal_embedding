from fastai.vision import *
from fastai.metrics import error_rate
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
from torch.utils import data
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import configparser
from pymagnitude import *
import argparse
import subprocess
import time

def build_dataframe(dict_fn, config, filter_mode = True):
    img_paths = []
    trans = []
    word_magnitude = None
    if filter_mode: 
        word_magnitude =  Magnitude(paths['word_magnitude']) 
        
    with open(paths['mmid_dir'] + "/" + dict_fn + "/index.tsv") as f:
        lines = f.read().splitlines()
        for line in lines:
            english_translation, index = line.split('\t')
            if filter_mode and english_translation not in word_magnitude: continue
            ends = ["01","02","03","04","05","06","07","08","09","10"]
            for img_num in ends:
                path =   dict_fn + "/" + str(index) + "/" + img_num + ".jpg"
                exists = os.path.isfile(paths['mmid_dir'] + "/" + path)
                # check to see whether the path exist
                if not exists: 
                    continue
                img_paths.append(path) 
                trans.append(english_translation)

    #dataframe with paths and translations
    df = pd.DataFrame({'paths': img_paths, 'trans' : trans}).dropna()
    df.to_csv("train_df.csv", sep='\t')

class SaveFeatures():
    features=None
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

def extract_features(train_data): 
    bs = 64 #batch size

    # default transformations applied below 
    my_data = ImageDataBunch.from_df(paths['code_dir'], train_data, valid_pct = 0, ds_tfms=get_transforms(), size=224, bs=bs, folder=paths['mmid_dir']).normalize(imagenet_stats)
    learn = cnn_learner(my_data, models.resnet50, metrics=error_rate)

    #This changes the forwards layers of the model 
    learn.fit_one_cycle(2)

    sf = SaveFeatures(learn.model[1][5]) 

    _= learn.get_preds(my_data.train_ds)

    return sf.features


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Image Feature Extraction')
    parser.add_argument("--config", type=str, default=None, help= 'config file to specify paths')
    parser.add_argument("--dict", type=str, default=None, help='image dictionary')
    parser.add_argument("--workers", type=int, default=5, help="number of processes to do work in the distributed cluster")
    parser.add_argument("--pid", type=int, default=None, help="set to true when just extracting features" )
    parser.add_argument("--mode",default="build", help="build: to build the df / create emb, eval: evaluate embeddings")

    params = parser.parse_args()
    # check params 
    assert os.path.isfile(params.config)
    config = configparser.ConfigParser()
    config.read(params.config)
    paths = config['PATHS']
    dict_fn = params.dict
    assert os.path.isdir(paths['mmid_dir'] + "/" + dict_fn)
    assert os.path.isfile(paths['word_magnitude'])
    assert params.mode == 'build' or params.mode == 'eval' or params.mode == 'partition'
    assert if params.mode == 'partition': params.pid != None
#    assert os.path.isfile('train_df.csv')
    if params.mode == 'build': 
        build_dataframe(dict_fn, paths) 
        # TODO: if everythings works then maybe don't save the file or delete
        # #build_dataframe(dict_fn, paths) 
        for process_id in range(params.workers): 
            cmd = ("qsub qrun.sh " + str(process_id) + " "  + str(params.workers) + " " + params.config + " " + params.dict).split()
            try:
                print("running " + str(process_id))
                subprocess.check_output(cmd)        
            except: 
                raise Exception("There was an error while running qsub to extract features")
            # TODO: sleeping to ensure load get to different machines, is there better way? 
            time.sleep(10) 
        print("Finished creating embeddings")
    elif params.mode == 'partition': 
        df_split = pd.read_csv('train_df.csv', sep='\t', index_col=[0])
        df_split = np.array_split(df_split, params.workers)[params.pid].reset_index().drop(columns=['index'])
        features = extract_features(df_split)
        translations = df_split['trans']
        filename = paths['data_dir'] + "img_embeddings_resnet50-" + str(params.pid) + ".txt"
        with open(filename, 'a') as f:
            for word, arr in zip(translations,features):
                f.write(word + "\t")
                np.savetxt(f, arr.reshape(1,len(arr)), delimiter=' ')
        exit(0)
    elif params.mode == 'eval': 
        df_split = pd.read_csv('train_df.csv', sep='\t', index_col=[0])
    else: 
        print("options are: eval, build, partition")
        exit(0)
        
        
    ##TODO Evaluate Image embeddings
    



