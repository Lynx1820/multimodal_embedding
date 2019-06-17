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

def build_dataframe(dict_fn, filter_mode = True):
    paths = []
    trans = []
    word_magnitude = None
    if filter_mode: 
        word_magnitude =  Magnitude(config['word_magnitude']) 
        
    with open(config['mmid_dir'] + "/" + dict_fn + "/index.tsv") as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            english_translations = line.split('\t')[1:]
            if filter_mode: 
                translation = None
                for word in english_translations:
                    if word in word_magnitude: 
                        translation = word
                        break 
                if translation == None: 
                    continue
            else: 
                translation = english_translations[0]
            ends = ["01","02","03","04","05","06","07","08","09","10"]
            for img_num in ends:
                path =  config['mmid_dir'] + "/" + dict_fn + "/" + str(i) + "/" + img_num + ".jpg"
                exists = os.path.isfile(path)
                # check to see whether the path exist
                if not exists: 
                    continue
                paths.append(path) 
                trans.append(translation)

    #dataframe with paths and translations
    df = pd.DataFrame({'paths': paths, 'trans' : trans}).dropna()
    df.to_csv("train_df.csv", sep='\t')

class saveFeatures():
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
    bs = 64
    my_data = ImageDataBunch.from_df("/nlp/users/dkeren", train_data, valid_pct = 0, ds_tfms=get_transforms(), size=224, bs=bs, folder="/nlp").normalize(imagenet_stats)
    learn = cnn_learner(my_data, models.resnet50, metrics=error_rate)

    #This changes the forwards layers of the model 
    learn.fit_one_cycle(2)

    sf = saveFeatures(learn.model[1][5]) 

    _= learn.get_preds(my_data.train_ds)

    return sf.features



config = configparser.ConfigParser()
config.read(sys.argv[2])
paths = config['PATHS']

full_df = pd.read_csv('/nlp/data/dkeren/train_df.csv', sep='\t', index_col=[0])
process_id = int(sys.argv[1])
df_split = np.array_split(full_df,100)[process_id]
df_split = df_split.reset_index()
df_split = df_split.drop(columns=['index'])
features = extract_features(df_split)
translations = df_split['trans']
filename = "/nlp/data/dkeren/img_embeddings_resnet50-" + str(process_id) + ".txt"
print("done")

for word, arr in zip(translations,features):
  with open(filename, 'a') as f:
      f.write(word + "\t")
      np.savetxt(f, arr.reshape(1,len(arr)), delimiter=' ')
