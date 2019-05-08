import numpy as np
from matplotlib import pyplot as plt
import torch
import jieba
from gensim.models import Word2Vec
dict_path = '../../data_hw6/dict.txt.big'
model_path = '../../data_hw6/word2vec_2.model'
from model import *

def problem_1():
    epoch = [1, 2, 3, 4, 5, 6, 7]
    train_acc = [0.72940, 0.75522, 0.77173, 0.79595, 0.82905, 0.86868, 0.90347]
    valid_acc = [0.74167, 0.74708, 0.74892, 0.74800, 0.74162, 0.73337, 0.72446]
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.title('Blue: training | Red: validation')
    plt.plot(epoch, train_acc, c= 'b')
    plt.plot(epoch, valid_acc, c= 'r')
    plt.savefig('img/p1.png')

def problem_2():
    epoch = [1, 2, 3, 4, 5, 6, 7]
    train_acc = [0.71599, 0.75474, 0.81954, 0.92183, 0.95768, 0.96636, 0.97091]
    valid_acc = [0.73225, 0.73358, 0.72646, 0.72375, 0.71267, 0.71700, 0.71404]
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.title('Blue: training | Red: validation')
    plt.plot(epoch, train_acc, c= 'b')
    plt.plot(epoch, valid_acc, c= 'r')
    plt.savefig('img/p2.png')

def problem_5_a():
    jieba.set_dictionary(dict_path)
    word_model = Word2Vec.load(model_path)
    sentence= ['在說別人白痴之前，先想想自己', '在說別人之前先想想自己,白痴']
    inputs = torch.zeros(2, 40, 300)
    for i in range(2):
        words_origin = list(jieba.cut(sentence[i]))
        vectors_origin= []
        for word in words_origin:
            try:
                vector = word_model[word]
                vectors_origin.append(vector)
            except:
                vectors_origin.append(word_model[' '])

        origin_len = len(vectors_origin)
        vectors = []
        remain = 40
        while remain > 0:
            if remain > origin_len:
                vectors += vectors_origin
                remain -= origin_len
            else:
                #start = random.randint(0, origin_len - remain) 
                vectors += vectors_origin[:remain]
                remain = 0

        inputs[i] = torch.from_numpy(np.array(vectors))
        model = get_rnn_model('B', 2)
        model.load_state_dict(torch.load('../../weights/0508_3.pkl'))
        out = model(inputs)
        print(out)



if __name__ == '__main__':
    problem_5_a()