import os
import argparse
import numpy as np
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('-path', help= 'Path to data folder', default= '../../data_pca')
args = parser.parse_args()


def get_images():
    image_name = [os.path.join(args.path, name) for name in os.listdir(args.path)]
    image_name.sort()
    images = np.zeros((len(image_name), 600, 600, 3))
    for i, name in enumerate(image_name):
        images[i] = np.array(io.imread(name))
    return images

def eigen(vectors):
    conv_matrix = (vectors) @ vectors.T
    eigen_values, eigen_vectors = np.linalg.eig(conv_matrix)
    return eigen_values, eigen_vectors
    
def show_image(vector):
    image = vector.reshape((600, 600, 3)).copy()
    image -= np.min(image)
    image /= np.max(image)
    image = (image * 255).astype(np.uint8)
    io.imshow(image)
    io.show()

def main():
    images = get_images()
    vectors = images.reshape((images.shape[0], -1))# (415, 1080000)
    mean = np.mean(vectors, 0)
    for i in range(len(mean)):
        vectors[:, i] -= mean[i] 
    eigen_values, eigen_vectors = eigen(vectors)

    param = []
    faces  = []
    target = vectors[400]
    for i in range(20):
        eig_v = eigen_vectors[:, i].copy()
        eig_face = (vectors.T) @ eig_v 
        eig_face /= np.sqrt(np.sum(eig_face @ eig_face))
        param.append(target @ eig_face)
        faces.append(eig_face)
    
    show_image(mean + target)
    show_image(mean)
  
    recons = mean
    for i in range(len(param)):
        recons += param[i] * faces[i]
    show_image(recons)
    
def test():
    images = get_images()
    
    
if __name__ == '__main__':
    main()