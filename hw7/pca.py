import os
import argparse
import numpy as np
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default= '../../data_pca')
parser.add_argument('-original', default= None)
parser.add_argument('-output', default= None)
args = parser.parse_args() 
dataset_path = args.dataset

def get_images():
    image_name = [os.path.join(dataset_path, name) for name in os.listdir(dataset_path)]
    image_name.sort()
    images = np.zeros((len(image_name), 600, 600, 3))
    for i, name in enumerate(image_name):
        images[i] = np.array(io.imread(name))
    return images

def eigen(vectors):
    conv_matrix = (vectors) @ vectors.T
    eigen_values, eigen_vectors = np.linalg.eig(conv_matrix)
    return eigen_values, eigen_vectors

def vector2image(vector):
    image = vector.reshape((600, 600, 3)).copy()
    image -= np.min(image)
    image /= np.max(image)
    image = (image * 255).astype(np.uint8)
    return image

def show_image(vector):
    image = vector2image(vector)
    io.imshow(image)
    io.show()

def plot_eigenfaces():
    images = get_images()
    vectors = images.reshape((images.shape[0], -1))# (415, 1080000)
    mean = np.mean(vectors, 0)
    mean_face = vector2image(mean)
    io.imsave('images/mean.png', mean_face)
    for i in range(len(mean)):
        vectors[:, i] -= mean[i] 
    eigen_values, eigen_vectors = eigen(vectors)
    
    for i in range(10):
        eig_vector = eigen_vectors[:, i]
        eig_face = (vectors.T) @ eig_vector 
        image = vector2image(eig_face)
        io.imsave('images/eigen_{}.png'.format(i+1), image)

def reconstruct_face(face, meanface, eigenfaces):
    #　eigenfaces are normalized
    weights= []
    for eigenface in eigenfaces:
        weights.append(eigenface @ face)
    
    recons = meanface
    for i in range(len(eigenfaces)):
        recons += weights[i] * eigenfaces[i]
    image = vector2image(recons)
    return image   

def reconstruct_faces():
    images = get_images()
    vectors = images.reshape((images.shape[0], -1))# (415, 1080000)
    mean = np.mean(vectors, 0)
    for i in range(len(mean)):
        vectors[:, i] -= mean[i] 
    eigen_values, eigen_vectors = eigen(vectors)

    eigenfaces  = []
    for i in range(5):
        eig_v = eigen_vectors[:, i].copy()
        eig_face = (vectors.T) @ eig_v 
        eig_face /= np.sqrt(np.sum(eig_face ** 2))
        eigenfaces.append(eig_face)
    
    targets = [0, 119, 200, 300, 400]
    for i, target_id in enumerate(targets):
        target_face = vectors[target_id]
        io.imsave('images/{}_original.png'.format(i+1), vector2image(mean + target_face))
        reconstruct = reconstruct_face(target_face, mean, eigenfaces)
        io.imsave('images/{}_reconstruct.png'.format(i+1), reconstruct)

def test():
    images = get_images()
    vectors = images.reshape((images.shape[0], -1))# (415, 1080000)
    mean = np.mean(vectors, 0)
    for i in range(len(mean)):
        vectors[:, i] -= mean[i] 
    eigen_values, eigen_vectors = eigen(vectors)
    print(eigen_values[:10])
    eigen_values = np.sqrt(np.abs(eigen_values))
    eigen_values /= np.sum(eigen_values)
    print(eigen_values[: 5])

def main():
    images = get_images()
    vectors = images.reshape((images.shape[0], -1))# (415, 1080000)
    mean = np.mean(vectors, 0)
    for i in range(len(mean)):
        vectors[:, i] -= mean[i] 
    eigen_values, eigen_vectors = eigen(vectors)

    eigenfaces  = []
    for i in range(5):
        eig_v = eigen_vectors[:, i].copy()
        eig_face = (vectors.T) @ eig_v 
        eig_face /= np.sqrt(np.sum(eig_face ** 2))
        eigenfaces.append(eig_face)

    origin_img = np.array(io.imread(args.original)).reshape((-1))
    recons_img = reconstruct_face(origin_img - mean, mean, eigenfaces)
    io.imsave(args.output, recons_img)
    
if __name__ == '__main__':
    # main()
    test()