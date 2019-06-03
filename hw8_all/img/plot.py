import numpy as np
from matplotlib import pyplot as plt

def problem2():
    parameter_num = [31751, 43175, 50455, 63551, 106791, 38087]
    acc = [0.5, 0.596, 0.603, 0.593, 0.636, 0.62661]
    plt.xlabel('Parameter number')
    plt.ylabel('Accuracy')
    plt.scatter(parameter_num, acc)
    plt.savefig('problem2.png')

def problem3():
    parameter_num = ['40791', '92663', '245319', '353559', '21116615']
    acc = [0.621, 0.644, 0.648, 0.636, 0.69]
    plt.xlabel('Parameter number')
    plt.ylabel('Accuracy')
    plt.scatter(parameter_num, acc)
    plt.savefig('problem3.png')

if __name__ == '__main__':
    problem3()