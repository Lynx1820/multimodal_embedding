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

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

class my_dataset(data.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, img_path, filenames, labels):
        """ Intialize the dataset
        """
        self.labels = labels
        self.filenames = filenames
        self.img_path = img_path
        self.len = len(self.filenames)
        
    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        #print(self.img_path + self.filenames[index])
        img = Image.open(self.img_path + "/" + self.filenames[index]).convert('RGB') 
        return Variable(normalize(to_tensor(scaler(img))))

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    


#base path to images
img_path = '/nlp/data/MMID/raw'
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
                for translation in english_translations:
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
    df = pd.DataFrame({'paths': paths, 'trans' : trans})
    df.to_csv("train_df.csv", sep='\t')

def extract_features(train_data): 
    bs = 64
    data = ImageDataBunch.from_df("/", train_data, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)

    #This changes the forwards layers of the model 
    learn.fit_one_cycle(1)

    # Get the mapping of parameter names to values from the trained Learner and a new ResNet.
    learner_state_dict = learn.model.state_dict()
    resnet = models.resnet34(pretrained=False)
    resnet_state_dict = resnet.state_dict()

    # Create a new state dictionary with the keys from the ResNet and the values from the Learner.
    state_dict = {}
    for learner_key, resnet_key in list(zip(learner_state_dict.keys(), resnet_state_dict.keys())):
        state_dict[resnet_key] = learner_state_dict[learner_key]

    # Hacky way to fix the shape of these parameters in the last layer, which will be removed.
    state_dict['fc.weight'] = torch.randn([1000, 512])
    state_dict['fc.bias'] = torch.randn([1000]) 

    # Load the updated state dictionary into the new ResNet and then create a new model with the last layer removed.
    resnet.load_state_dict(state_dict)
    removed = list(resnet.children())[:-1]
    model = torch.nn.Sequential(*removed)

    # Get convolutional features from model with the fully connected layer removed.
    dataset = my_dataset(str(img_path), df['paths'],df['trans'])
    dataloader = data.DataLoader(dataset, batch_size = bs, num_workers = 1)

    features = []
    init = 1
    for _ , sample_batched in enumerate(dataloader):
        #print(sample_batched.shape) #64 x 3 x 224 x 224
        torch_features = model.forward(sample_batched) 
        tmp = torch_features.data.numpy() 
        numpy_features = tmp.reshape(tmp.shape[:2])
        if init == 1:
            features = numpy_features
            init = 0
        else: 
            features = np.append(features, numpy_features,axis = 0)
    return features


#load dataframe or create dataframe
df = pd.read_csv('/home1/d/dkeren/599/multimodal_embedding/train_df.csv', sep='\t', index_col=[0])
df = df.dropna()
#rows = df.shape[0]
process_id = sys.argv[1]
df_split = np.array_split(df,100)[process_id]
fea = extract_features(df_split)
filename = "img_embeddings_resnet34.npz" + str(process_id)
np.savez(filename, df_split['trans'], fea)
