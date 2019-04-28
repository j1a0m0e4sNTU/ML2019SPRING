import pandas as pd
import numpy as np
import jieba
jieba.set_dictionary('../../data_hw6/dict.txt.big')
from gensim.models import Word2Vec

x_train_path = '../../data_hw6/train_x.csv'
y_train_path = '../../data_hw6/train_y.csv'
x_test_path  = '../../data_hw6/test_x.csv'
word2vec_model_path = '../../data_hw6/word2vec.model'

def load_label(path):
    csv = pd.read_csv(path)
    label = np.array(csv['label'])
    return label

def save_word2vec():
    csv = pd.read_csv(x_train_path)
    raw_sentences = np.array(csv['comment'])
    cut_all = []
    
    for i, sentence in enumerate(raw_sentences):
        cut_list = list(jieba.cut(sentence))
        cut_all.append(cut_list)

    model = Word2Vec(cut_all, size= 300, window= 5)
    model.save(word2vec_model_path)

def test():
    print('- test -')
    model = Word2Vec.load(word2vec_model_path)
    

if __name__ == '__main__':
    save_word2vec()
