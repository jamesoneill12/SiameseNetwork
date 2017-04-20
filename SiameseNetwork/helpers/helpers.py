import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import os

def retrieve_embeddings():
    path = "C:/Users/1/James/grctc/GRCTC_Project/Classification/Data/Embeddings/Glove/twitter/glove.twitter.27B.50d.txt"

def get_sentence_pairs(path=''):
    if path is '':
        root = "C:/Users/1/James/Research/Projects/DTW_Similarity_Project/Datasets/Sentence Similarity/dataset-sts/data/sts/sick2014/"

    trainpath = "SICK_train.txt"
    testpath = "SICK_test_annotated.txt"

    train = pd.read_csv(root + trainpath, sep="\t")
    test = pd.read_csv(root + testpath, sep="\t")
    train_sentenceA = [line.split() for line in train['sentence_A'].tolist()]
    train_sentenceB = [line.split() for line in train['sentence_B'].tolist()]
    test_sentenceA = [line.split() for line in test['sentence_A'].tolist()]
    test_sentenceB = [line.split() for line in test['sentence_B'].tolist()]
    y_train = train['relatedness_score']
    y_test = test['relatedness_score']

    params = {'train_sent_A':train_sentenceA, 'train_sent_B':train_sentenceB,
              'test_sent_A':test_sentenceA,'test_sent_B':test_sentenceB,
              'y_train':y_train, 'y_test':y_test}

    return params

def w2v(params):
    train_sents = list(ta+tb for (ta,tb) in zip(params['train_sent_A'],params['train_sent_B']))
    test_sents = list(ta+tb for (ta,tb) in zip(params['test_sent_A'],params['test_sent_B']))
    all_sents = train_sents+test_sents
    word2vec = Word2Vec(all_sents, size=100, window=4, min_count=10, workers=4)
    word2vec.init_sims(replace=True)
    word_vectors = [word2vec[word] for word in word2vec.vocab]
    word2vec.save(os.getcwd() + 'word2vec.pkl')
    print('Found %s texts.' % len(all_sents))
    return word_vectors

