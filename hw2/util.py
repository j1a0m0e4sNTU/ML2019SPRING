import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_raw_data(path):
    file = np.genfromtxt(path, delimiter= ',')
    data = file[1:]
    return data

def get_total_feature(csv_file, provided_feature):
    provided = get_raw_data(provided_feature)
    csv = pd.read_csv(csv_file)
    h, w = provided.shape

    total = np.empty((h, w+1), dtype= np.float)
    total[:, 0] = csv['education_num']
    total[:, 1:] = provided
    return total

def normalize_feature(feature, mean, std):
    mean[3] = 0
    mean[7:]= 0
    std[3]  = 1
    std[7:] = 1
    normalized = (feature - mean) / std
    return normalized

def get_train_valid_data(total_x, total_y, fold= 0):
    fold = fold % 5
    total_size = total_x.shape[0]
    valid_size = total_size // 5

    valid_sampler = np.arange(valid_size * fold, valid_size * (fold + 1))
    train_sampler = np.arange(total_size - valid_size)
    train_sampler[valid_size * fold:] += valid_size

    train_x, train_y = total_x[train_sampler], total_y[train_sampler]
    valid_x, valid_y = total_x[valid_sampler], total_y[valid_sampler]
    return train_x, train_y, valid_x, valid_y

def test():
    total_x = get_raw_data('data/X_train')
    total_y = get_raw_data('data/Y_train')
    train_x, train_y, valid_x, valid_y = get_train_valid_data(total_x, total_y, fold= 0)
    print(type(total_x))

def test2():
    x = get_total_feature('data/train.csv', 'data/X_train')
    m = np.mean(x, 0)
    s = np.std(x, 0)
    n = normalize_feature(x, m ,s)
    print(n[:10])
    print(n.shape)


def get_objs(array):
        obj_list = []
        for i in array:
            if i in obj_list:
                continue
            obj_list.append(i)
        return obj_list

if __name__ == '__main__':
    test2()