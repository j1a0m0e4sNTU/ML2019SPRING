import pandas as pd
import numpy as np
import jieba
jieba.set_dictionary('../../data_hw6/dict.txt.big')
from gensim.models import Word2Vec
import torch
from torch.utils.data import Dataset

x_train_path = '../../data_hw6/train_x.csv'
y_train_path = '../../data_hw6/train_y.csv'
x_test_path  = '../../data_hw6/test_x.csv'
word2vec_model_path = '../../data_hw6/word2vec.model'

def load_label(path):
    csv = pd.read_csv(path)
    label = np.array(csv['label'])
    return label

def save_word2vec():
    train_csv = pd.read_csv(x_train_path)
    test_csv  = pd.read_csv(x_test_path)
    raw_sentences = np.append(np.array(train_csv['comment']), np.array(test_csv['comment'])) 
    cut_all = []
    
    for i, sentence in enumerate(raw_sentences):
        cut_list = list(jieba.cut(sentence))
        cut_all.append(cut_list)

    model = Word2Vec(cut_all, size= 300, window= 5, iter= 10, sg= 0)
    model.save(word2vec_model_path)    

class WordsData(Dataset):
    def __init__(self, mode= 'train', x_path= x_train_path, y_path= y_train_path, model_path= word2vec_model_path):
        super().__init__()
        self.mode = mode
        x_csv = pd.read_csv(x_path)
        self.x_data = np.array(x_csv['comment'])
        self.y_data = None
        if mode == 'train':
            y_csv = pd.read_csv(y_path)
            self.y_data = np.array(y_csv['label'])
    
        self.model = Word2Vec.load(model_path)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        sentence = self.x_data[index]
        words = list(jieba.cut(sentence))
        vectors = []
        for word in words:
            vector = None
            try:
                vector = self.model[word]
            except:
                vector = self.model[' ']
            vectors.append(vector)

        vectors = torch.from_numpy(np.array(vectors))

        if self.mode == 'train':
            label = torch.LongTensor([self.y_data[index]])
            return vectors, label
        else:
            return vectors

        

def test():
    print('test')
    dataset = WordsData(mode= 'train')
    for i in range(10):
        vectors, label = dataset[i]
        print("vector size: {} | label: {}".format(vectors.size(), label))
    
    

if __name__ == '__main__':
    test()
