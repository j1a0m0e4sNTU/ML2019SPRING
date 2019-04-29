import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec
import torch
from torch.utils.data import Dataset, DataLoader

dict_path = '../../data_hw6/dict.txt.big'
x_train_path = '../../data_hw6/train_x.csv'
y_train_path = '../../data_hw6/train_y.csv'
x_test_path  = '../../data_hw6/test_x.csv'
word2vec_model_path = '../../data_hw6/word2vec.model'

def load_label(path):
    csv = pd.read_csv(path)
    label = np.array(csv['label'])
    return label

def save_word2vec():
    jieba.set_dictionary(dict_path)
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
    def __init__(self, mode= 'train', x_path= x_train_path, y_path= y_train_path, model_path= word2vec_model_path, dict_path= dict_path, seq_len= 30):
        super().__init__()
        self.mode = mode
        x_csv = pd.read_csv(x_path)
        self.x_data = np.array(x_csv['comment'])
        self.y_data = None
        cut_size = int(len(self.x_data) * 0.8)
        
        if mode == 'train':
            self.x_data = self.x_data[:cut_size]
            y_csv = pd.read_csv(y_path)
            self.y_data = np.array(y_csv['label'])[:cut_size]
        elif mode == 'valid':
            self.x_data = self.x_data[cut_size:]
            y_csv = pd.read_csv(y_path)
            self.y_data = np.array(y_csv['label'])[cut_size:]
    
        self.model = Word2Vec.load(model_path)
        jieba.set_dictionary(dict_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        sentence = self.x_data[index]
        words_origin = list(jieba.cut(sentence))
        words_length = len(words_origin)
        words = []
        remain = self.seq_len
        while remain > 0:
            if remain > words_length:
                words += words_origin
                remain -= words_length
            else: 
                words += words_origin[:remain]
                remain = 0
        
        vectors = []
        for word in words:
            vector = None
            try:
                vector = self.model[word]
            except:
                vector = self.model[' ']
            vectors.append(vector)
        vectors = torch.from_numpy(np.array(vectors))
        
        if self.mode in ['train', 'valid']:
            label = torch.Tensor([self.y_data[index]])
            return vectors, label
        else:
            return vectors

        

def test():
    print('test')
    words = WordsData('train')
    data = DataLoader(words, batch_size= 4)
    for i, (vector, label) in enumerate(data):
        print(vector.size(), label.size())
        if i == 10:
            break

if __name__ == '__main__':
    test()
