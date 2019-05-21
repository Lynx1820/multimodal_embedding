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

#base path to images
img_path = '/nlp'
dict_path = '/nlp/data/MMID/dictionaries' 


dictionaries = {"dict.es":"scale-spanish-package", "dict.fr":"scale-french-package", "dict.id":"scale-indonesian-package", "dict.it":"scale-italian-package", "dict.nl":"scale-dutch-package"}
def build_dataframe():
    # All 5 dictionaries
    paths = []
    trans = []
    for dict_fn in tqdm(dictionaries):
        with open(str(dict_path) + "/" + dict_fn) as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                english_translations = line.split('\t')[1:]
                if "(name)" in english_translations: 
                    continue
                # for every translation, we ge the image path_id - which will then extract features for 
                #for translation in english_translations:
                translation = english_translations[0] #since the image is the same, we just train on the first word
                ends = ["01","02","03","04","05","06","07","08","09","10"]
                for img_num in ends:
                    path = str(img_path) + "/" + dictionaries[dict_fn] + "/" + str(i) + "/" + img_num + ".jpg"
                    exists = os.path.isfile(path)
                    # check to see whether the path exist
                    if not exists: 
                        print("Could not find file: " +  path)
                        continue
                    paths.append(path) 
                    trans.append(translation)

    #dataframe with paths and translations
    df = pd.DataFrame({'paths': paths, 'trans' : trans}).dropna()
    df.to_csv("train_df.csv", sep='\t')

# this is a hook (learned about it here: https://forums.fast.ai/t/how-to-find-similar-images-based-on-final-embedding-layer/16903/13)
# hooks are used for saving intermediate computations
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
    bs = 64
    my_data = ImageDataBunch.from_df("/nlp/users/dkeren", train_data, valid_pct = 0, ds_tfms=get_transforms(), size=224, bs=bs, folder="/nlp").normalize(imagenet_stats)
    learn = cnn_learner(my_data, models.resnet34, metrics=error_rate)

    #This changes the forwards layers of the model 
    learn.fit_one_cycle(4)

    sf = SaveFeatures(learn.model[1][5]) ## Output before the last FC layer
    ## By running this feature vectors would be saved in sf variable initated above
    _= learn.get_preds(my_data.train_ds)
    _= learn.get_preds(DatasetType.Valid)

    #img_paths = [str(x) for x in (list(my_data.train_ds.items))]
    #feature_dict = dict(zip(trans,sf.features))
    #img_dict = dict(zip(trans,img_path))
    return sf.features


#load dataframe or create dataframe
full_df = pd.read_csv('/nlp/data/dkeren/train_df.csv', sep='\t', index_col=[0])
#rows = df.shape[0]
process_id = int(sys.argv[1])
df_split = np.array_split(full_df,100)[process_id]
#print( df_split.shape[0]) 
#print("done splitting data")
df_split = df_split.reset_index()
df_split = df_split.drop(columns=['index'])
fea = extract_features(df_split)

filename = "/nlp/data/dkeren/img_embeddings_resnet34-" + str(process_id) + ".npz"
# print("done")
# np.savez(filename, df_split['trans'], fea)
