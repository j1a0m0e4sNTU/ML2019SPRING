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

def normalize_min_max(feature):
    f_min = np.min(feature, axis= 0)
    f_max = np.max(feature, axis= 0)
    normalized = (feature - f_min)/ (f_max - f_min)
    return normalized

def discretalize(feature, idx):
    # the input feature is already normalized
    data_num, f_num = feature.shape
    feature_new = np.zeros((data_num, f_num + 6), dtype= np.float)
    f_discrete = np.zeros((data_num, 6))
    f_origin = feature[:, idx]

    f_discrete[:, :] = [1, 0, 0, 0, 0, 0]
    f_discrete[f_origin > -2] = [0, 1, 0, 0, 0, 0]
    f_discrete[f_origin > -1] = [0, 0, 1, 0, 0, 0]
    f_discrete[f_origin > 0]  = [0, 0, 0, 1, 0, 0]
    f_discrete[f_origin > 1]  = [0, 0, 0, 0, 1, 0]
    f_discrete[f_origin > 2]  = [0, 0, 0, 0, 0, 1]

    feature_new[:, : f_num] = feature
    feature_new[:, f_num : f_num + 6] = f_discrete
    
    return feature_new

def discretalize_all(feature):
    continuous_id = [0, 1, 2, 4, 5, 6]
    for i in continuous_id:
        feature = discretalize(feature, i)
    
    return feature

def add_constant_column(feature):
    h, w = feature.shape
    feature_new = np.ones((h, w + 1))
    feature_new[:, 1:] = feature
    return feature_new

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
    x = normalize_feature(x, m ,s)
    x = discretalize_all(x)
    print(x.shape)


def get_objs(array):
        obj_list = []
        for i in array:
            if i in obj_list:
                continue
            obj_list.append(i)
        return obj_list

if __name__ == '__main__':
    test2()