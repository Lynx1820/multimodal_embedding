"""
Purpose:
- Evaluate multimodal embeddings against 6 word similarity tests
"""
from process_eval_set import get_eval_set_dict
from argparse import ArgumentParser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from scipy import stats
import os
import configparser
from pymagnitude import *
import pickle 

def convert_dict_to_txt(path):
    """
    convert the dictionary that contains all predicted embeddings into txt format 
    """
    with open(path+'_all.p', 'rb') as fp:
        word_dict = pickle.load(fp)

    for k, v in word_dict.items():
        v = v.astype('<U100')
        v = np.insert(v, 0, k)
        with open(path+'_all.txt', 'a') as f:
            np.savetxt(f, v.reshape(1, v.shape[0]), fmt='%s')

def compute_pair_sim(word1, word2): 
    """
    compute cosine similarity between two words
    """
    dot_product = np.dot(word1, word2)
    length_word1 = np.linalg.norm(word1)
    length_word2 = np.linalg.norm(word2)
    return dot_product/(length_word1 * length_word2)

def compute_sim(eval_set, word_dict):
    """
    compute similarity for all words in the evaluation set
    @param word_dict: dictionary: keys: words, values: learned embeddings
    @param eval_set_type: type of evaluation set (zs/vis) 
    @return a numpy array of word similarity
    """ 
    word_sim = []
    for i in range(eval_set.shape[0]):
        embedding1 = word_dict[eval_set[i][0]]
        embedding2 = word_dict[eval_set[i][1]]
        pair_sim = compute_pair_sim(embedding1, embedding2)
        word_sim.append(pair_sim)
    word_sim = np.asarray(word_sim)
    
    return word_sim

def compute_sim_magnitude(eval_set, word_dict):
    """
    compute similarity for all words in the evaluation set
    @param word_dict: dictionary: keys: words, values: learned embeddings
    @param eval_set_type: type of evaluation set (zs/vis) 
    @return a numpy array of word similarity
    """ 
    word_sim = []
    for i in range(eval_set.shape[0]):
        word1 = eval_set[i][0]
        word2 = eval_set[i][1]
        emb1 = [word_dict.query(word1)]
        emb2 = [word_dict.query(word2)]
        for label in range(1,10):
            tmp1 = word1 + "_" + str(label) 
            tmp2 = word2 + "_" + str(label)
            if tmp1 in word_dict: 
                emb1.append(word_dict.query(tmp1))
            if tmp2 in word_dict: 
                emb2.append(word_dict.query(tmp2))
        pair_sim = sum(np.amax(cosine_similarity(np.array(emb1), np.array(emb2)), axis=1))/len(emb1)
        word_sim.append(pair_sim)
    word_sim = np.asarray(word_sim)
    
    return word_sim

def evaluate(eval_set_type, word_dict, dict_format, eval_set_dict, paths):
    """
    Print out evaluation results (correlation, P-value) for all sets, either of type ZS or VIS 
    @param eval_set_type: Type of eval set (ZS/VIS)
    @param word_dict: corresponding dictionary: keys: zs/vis words, values: predicted embeddings 
    """ 
    path = paths['eval_dir']
    for eval_name, eval_set in eval_set_dict.items():
        eval_set = pd.read_csv(path+"/"+str(eval_name)+'_'+eval_set_type+'.txt', sep=' ', header=None).as_matrix()
        if dict_format == 'dict':
            model_sim = compute_sim(eval_set, word_dict)
        else:
            model_sim = compute_sim_magnitude(eval_set, word_dict)
        cor, pval = stats.spearmanr(model_sim, eval_set[:,2])
        print("Correlation for {} ({}): {:.3f}, P-value: {:.3f}".format(eval_name, eval_set_type, cor, pval))
    print()

def evaluate_all(eval_set_dict, word_dict, dict_format):
    """
    Print out evaluation results (correlation, P-value) for all sets (full set)
    @param eval_set_dict: List of eval sets (in matrix form)
    @param word_dict: corresponding dictionary: keys: all words, values: predicted embeddings 
    @param dict_format: format of dictionary, 'dict' (normal dictionary) or 'magnitude' (magnitude object)
    """  
    print("Evaluation List Used: wordsim_sim, wordsim_rel, simlex, men")
    for eval_name, eval_set in eval_set_dict.items():
        if dict_format == 'dict':
            model_sim = compute_sim(eval_set, word_dict)
        else:
            model_sim = compute_sim_magnitude(eval_set, word_dict)
        cor, pval = stats.spearmanr(model_sim, eval_set[:,2])
        print("Correlation for {} (all): {:.3f}, P-value: {:.3f}".format(eval_name, cor, pval))
    print() 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model', default='normal', type=str, help='[normal, c_linear, c_neural]')
    parser.add_argument('path', type=str, help='path to dictionary that contains predicted embeddings')
    parser.add_argument("-c", type=str, help="path to config file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.c == None:
        raise Exception("Need config file")
    config = configparser.ConfigParser()
    config.read(args.c)
    paths = config['PATHS']
    
    # load evaluation sets
    eval_set_dict = get_eval_set_dict(paths)
    
    #args.m: linear, neural, c_linear, c_neural
    # evaluate a normal model (not concatenated)
    if args.model == 'normal':
        # open dictionaries that contains predicted embeddings
        with open(args.path+"_vis.p", 'rb') as fp:
            word_dict_vis = pickle.load(fp)
        with open(args.path+"_zs.p", 'rb') as fp:
            word_dict_zs = pickle.load(fp)
        with open(args.path+"_all.p", 'rb') as fp:
            word_dict_all = pickle.load(fp)
        evaluate('vis', word_dict_vis, 'dict', eval_set_dict, paths)
        evaluate('zs', word_dict_zs, 'dict', eval_set_dict, paths)
        evaluate_all(eval_set_dict, word_dict_all, 'dict')
    # evaluate a concatenated model
    elif args.model == 'word_mode': 
        word_dict = Magnitude(paths['word_magnitude'])
        print("Using word magnitude file from: " + paths['word_magnitude'] )
        evaluate('vis', word_dict, 'magnitude', eval_set_dict, eval_set_dict, paths)
        evaluate('zs', word_dict, 'magnitude', eval_set_dict, eval_set_dict, paths)
        evaluate_all(eval_set_dict, word_dict, 'magnitude')
    elif args.model == 'img_mode': 
        img_dict = Magnitude(paths['image_magnitude'])
        print("Using image magnitude file from: " + paths['image_magnitude'] )
        evaluate('vis', img_dict, 'magnitude', eval_set_dict, paths)
        evaluate('zs', img_dict, 'magnitude', eval_set_dict, paths)
        evaluate_all(eval_set_dict, img_dict, 'magnitude')
    elif args.model == 'c_linear' or args.model == 'c_neural':    
        word_dict = Magnitude(paths['word_magnitude'])
        if args.model == 'c_linear':        
            pred_dict = Magnitude(paths['lin_magnitude'])
        else:        
            pred_dict = Magnitude(paths['nn_magnitude'])
        # concatenate a model with glove 
        fused_dict = Magnitude(word_dict, pred_dict)
        print('Dimension of concatenated vectors: {}'.format(fused_dict.dim))

        evaluate('vis', fused_dict, 'magnitude', eval_set_dict, paths)
        evaluate('zs', fused_dict, 'magnitude', eval_set_dict, paths)
        evaluate_all(eval_set_dict, fused_dict, 'magnitude')

if __name__ == '__main__':
    # convert dictionary to txt file, then convert to Magnitude format in command line 
    # comment out main() if this function is called()
    args = parse_args()
    #convert_dict_to_txt('model/fasttext')
    main()

