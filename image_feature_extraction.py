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
from process_eval_set import get_eval_set_dict 
from evaluation import compute_pair_sim
from torch.utils.data import Dataset, DataLoader
from scipy import stats
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
            line = line.replace("\\t", "\t")
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

def extract_features(train_data): 
    bs = 64 #batch size

    # default transformations applied below 
    my_data = ImageDataBunch.from_df(paths['code_dir'], train_data, valid_pct = 0, ds_tfms=get_transforms(), size=224, bs=bs, folder=paths['mmid_dir']).normalize(imagenet_stats)
    learn = cnn_learner(my_data, models.resnet50, metrics=error_rate)

    #This changes the forwards layers of the model 
    learn.fit_one_cycle(5)

    sf = SaveFeatures(learn.model[1][5]) 

    _= learn.get_preds(my_data.train_ds)

    return sf.features


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Image Feature Extraction')
    parser.add_argument("--config", type=str, default=None, help= 'config file to specify paths')
    parser.add_argument("--dict", type=str, nargs='+', default=None, help='image dictionary')
    parser.add_argument("--workers", type=int, default=5, help="number of processes to do work in the distributed cluster")
    parser.add_argument("--pid", type=int, default=None, help="set to true when just extracting features" )
    parser.add_argument("--mode",choices=['build', 'eval', 'partition', 'merge'], help="build: to build the df / create emb, eval: evaluate embeddings")

    params = parser.parse_args()
    # check params 
    assert os.path.isfile(params.config)
    config = configparser.ConfigParser()
    config.read(params.config)
    paths = config['PATHS']
    for dict_fn in params.dict:
        assert os.path.isdir(paths['mmid_dir'] + "/" + dict_fn)
    ## TODO: assert for all partials workers
    assert os.path.isfile(paths['word_magnitude'])
    assert params.mode == 'build' or params.mode == 'eval' or params.mode == 'partition'
#    assert os.path.isfile('train_df.csv')
    # TODO: suppprt more dictionaries
    if params.mode == 'build': 
        # TODO: if everythings works then maybe don't save the file or delete
        for dict_fn in params.dict: 
            build_dataframe(dict_fn, paths) 
            for process_id in range(params.workers): 
                cmd = ("qsub qrun.sh " + str(process_id) + " "  + str(params.workers) + " " + params.config + " " + dict_fn).split()
                try:
                    print("running " + str(process_id))
                    subprocess.check_output(cmd)        
                except: 
                    raise Exception("There was an error while running qsub to extract features")
                # TODO: sleeping to ensure load get to different machines, is there better way? 
                time.sleep(10) 
        print("Finished creating embeddings")
    elif params.mode == 'partition': 
        df_split = pd.read_csv(params.dict[0] + '-train_df.csv', sep='\t', index_col=[0])
        df_split = np.array_split(df_split, params.workers)[params.pid].reset_index().drop(columns=['index'])
        features = extract_features(df_split)
        translations = df_split['trans']
        filename = paths['data_dir'] + "/img_embeddings_resnet50-" + params.dict[0] + "-" + str(params.pid) + ".txt"
        with open(filename, 'a') as f:
            for word, arr in zip(translations,features):
                f.write(word + "\t")
                np.savetxt(f, arr.reshape(1,len(arr)), delimiter=' ')
        exit(0)
    # TODO: Handle the different 
    elif params.mode == 'merge': 
        conc_files = {}
        for dict_fn in params.dict:
            files = [] 
            tmp_file = paths['data_dir'] + "/full_img_embeddings_resnet50-" + dict_fn + ".txt"
            for pid in range(params.workers):
                curr_file = paths['data_dir'] + "/img_embeddings_resnet50-" + dict_fn + "-" + str(pid) + ".txt"
                files.append(curr_file)
            conc_files[tmp_file] = files
        fn = paths['data_dir'] + "/full_img_embeddings_resnet50.txt"
        magnitude_fn = paths['data_dir'] + "/full_img_embeddings.magnitude"
        full_file = open(fn, 'w')
        words = set()
        for temp, partial_files in conc_files.items():
            full_dict_file = open(temp, 'w')
            for filename in partial_files: 
                with open(filename, 'r') as file: 
                    for line in file.read():
                        words.add(line.split('\t')[0])
                        full_dict_file.write(line)
                        full_file.write(line)
            full_dict_file.close()
        full_file.close()
        print("done writing new dictionaries")
        cmd = ("python -m pymagnitude.converter -i " + fn +  " -o "  + magnitude_fn).split()

        try: 
            print("running: " + str(cmd))
            subprocess.check_output(cmd)       
        except: 
            raise Exception("There was an error running" + str(cmd))
    elif params.mode == 'eval': 
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
                    cos_sim = compute_pair_sim(embeddings.query(word1), embeddings.query(word2))
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
        
    ##TODO Evaluate Image embeddings
    



