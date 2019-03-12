import numpy as np

def get_raw_data(path):
    file = np.genfromtxt(path, delimiter= ',')
    data = file[1:]
    return data

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
    print(total_x.shape[0])
    print(train_x.shape)
    print(train_y.shape)
    print(valid_x.shape)
    print(valid_y.shape)

if __name__ == '__main__':
    test()