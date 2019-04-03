import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_train_cnn():
    file = pd.read_csv('results/vgg_cc_2.csv')
    epochs = np.array(file['epoch'])
    train_acc = np.array(file['train_acc'])
    valid_acc = np.array(file['valid_acc'])
    
    plt.title('Training process of CNN method')
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.plot(epochs, train_acc, 'b')
    plt.plot(epochs, valid_acc, 'r')
    # plt.show()
    plt.savefig('doc/cnn.png')

def plot_train_dnn():
    file = pd.read_csv('results/dnn.csv')
    epochs = np.array(file['epoch'])
    train_acc = np.array(file['train_acc'])
    valid_acc = np.array(file['valid_acc'])
    
    plt.title('Training process of DNN method')
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.plot(epochs, train_acc, 'b')
    plt.plot(epochs, valid_acc, 'r')
    # plt.show()
    plt.savefig('doc/dnn.png')

def plot_cunfusion_matrix():
    train_file = pd.read_csv('../../data_hw3/train.csv')
    pred_file = pd.read_csv('predictions/pred_train.csv')
    ground_truth = np.array(train_file['label'])
    prediction = np.array(pred_file['label'])
    num = len(prediction)
    cut = int(0.8 * num)
    ground_truth = ground_truth[cut:]
    prediction = prediction[cut:]

    confusion = np.zeros((7, 7), dtype= np.float)
    for label in range(7):
        pred = prediction[ground_truth == label]
        count = len(pred)
        for i in range(7):
            confusion[label, i] = np.sum(pred == i)/count
    
    fig, ax = plt.subplots()
    ax.matshow(confusion)
    for (i, j), value in np.ndenumerate(confusion):
        ax.text(j, i, '{:0.2f}'.format(value), ha= 'center', va= 'center')
    # plt.show()
    plt.savefig('doc/confusion.png')

if __name__ == '__main__':
    print('- Plot -')
    plot_cunfusion_matrix()