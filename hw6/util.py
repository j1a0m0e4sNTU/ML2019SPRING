import sys
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
words_result_path = '../../data_hw6/words.txt'

important_words = ['回應','會爆','秀下限','瞎妹','ㄏㄏ','開口','邊緣人','森77','森七七','森氣氣','黑人問號',
                    '+1','廚餘','打臉','Hen棒','低能卡','被閃','甲','原po','原PO','啾咪','腦羞','打手槍',
                    '台男','台女','不意外','不ey','內射','中出','唱衰','噁爛','8+9','酸民','笑死','三小',
                    '乾你屁事','自以為','被嘴','約炮','傻B','馬的','肥宅','菜逼八','頗呵','台中','渣男',
                    '樓主','腦粉','氣pupu','八嘎囧','ㄊㄇ','D卡','幹你娘','仇女','有病',]

def adjust_jieba():
    for word in important_words:
        jieba.suggest_freq(word, True)

def get_cleaner_words(words):
    cleaner = []
    for word in words:
        if word[0] in ['b', 'B']:
            cleaner.append('B')
        else: 
            cleaner.append(word)
    return cleaner

def load_label(path):
    csv = pd.read_csv(path)
    label = np.array(csv['label'])
    return label

def save_word2vec():
    jieba.set_dictionary(dict_path)
    #adjust_jieba()
    train_csv = pd.read_csv(x_train_path)
    test_csv  = pd.read_csv(x_test_path)
    raw_sentences = np.append(np.array(train_csv['comment']), np.array(test_csv['comment'])) 
    cut_all = []
    file = open(words_result_path, 'w')
    for i, sentence in enumerate(raw_sentences):
        cut_list = list(jieba.cut(sentence))
        #cut_list = get_cleaner_words(cut_list)
        cut_all.append(cut_list)
        line = '{}\n'.format('/'.join(cut_list))
        file.write(line)

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
        #adjust_jieba()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        sentence = self.x_data[index]
        words_origin = list(jieba.cut(sentence))
        #words_origin = get_cleaner_words(words_origin)
        
        vectors_origin = []
        for word in words_origin:
            try:
                vector = self.model[word]
                vectors_origin.append(vector)
            except:
                vectors_origin.append(self.model[' '])
        
        origin_len = len(vectors_origin)
        vectors = []
        remain = self.seq_len
        while remain > 0:
            if remain > origin_len:
                vectors += vectors_origin
                remain -= origin_len
            else:
                #start = random.randint(0, origin_len - remain) 
                vectors += vectors_origin[:remain]
                remain = 0

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

def ensemble():
    test_num = 20000
    pred_all = np.zeros((test_num,))
    for i in range(5):
        file = pd.read_csv('predictions/{}.csv'.format(i))
        pred_all += file['label']
    
    count = 0
    file = open('predictions/e.csv', 'w')
    file.write('id,label\n')
    for i, score in enumerate(pred_all):
        pred = 1 if score > 2 else 0
        file.write('{},{}\n'.format(i, pred))

if __name__ == '__main__':
    if sys.argv[1] == '1':
        save_word2vec()
    else:
        ensemble()
