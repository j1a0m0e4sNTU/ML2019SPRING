import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import AutoEncoder
from dataset import *

class Manager():
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id = args.id
        self.dataset_path = args.dataset
        model = AutoEncoder(args.E, args.D)
        if args.load:
            model.load_state_dict(torch.load(args.load))
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.metric = nn.MSELoss()
        self.epoch_num = args.epoch
        self.batch_size = args.bs
        self.cluster_num = args.cluster_num
        self.save = args.save
        self.testcase = args.testcase
        self.csv = args.csv
        self.record_file = None
        self.best = {'epoch':0, 'loss': 999}
        self.test_images = get_test_image(args.dataset)

    def record(self, info):
        self.record_file.write(info + '\n')
        print(info)

    def train(self, data_train, data_valid):
        self.record_file = open('records/' + self.id + '.txt', 'w')
        for epoch in range(self.epoch_num):
            train_loss = 0
            self.model.train()
            for i, images in enumerate(data_train):
                images = images.to(self.device)
                images_out = self.model(images)
                self.optimizer.zero_grad()
                loss = self.metric(images_out, images)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            train_loss = train_loss / (i+1)
            valid_loss = self.get_valid_loss(data_valid)
            best_info = ""
            if (valid_loss < self.best['loss']):
                self.best['epoch'] = epoch
                self.best['loss'] = valid_loss
                torch.save(self.model.state_dict(), self.save)
                best_info = '* Best *'
            info = 'Epoch {:2d} | training loss: {:.5f} | validation loss: {:.5f} {}'.format(epoch+1, train_loss, valid_loss, best_info)
            self.record(info)
            self.plot()

    def get_valid_loss(self, data_valid):
        self.model.eval()
        valid_loss = 0
        for i, images in enumerate(data_valid):
            images = images.to(self.device)
            images_out = self.model(images)
            loss = self.metric(images_out, images)
            valid_loss += loss.item()
        return valid_loss / (i+1)

    def write_submission(self, predictions):
        file = open(self.csv, 'w')
        file.write('id,label\n')
        for i, pred in enumerate(predictions):
            file.write('{},{}\n'.format(i,pred))
        print('Finish submission: {}'.format(self.csv))

    def plot(self):
        images = self.test_images.to(self.device)
        images_out = self.model(images)
        plot_images(images_out, 'records/' + self.id + '.jpg')

    def get_vectors(self, data):
        vector_all = None
        for i, images in enumerate(data):
            images = images.to(self.device)
            vector = self.model.encode(images)
            vector = vector.cpu().detach()
            if i == 0:
                vector_all = vector
            else:
                vector_all = torch.cat([vector_all, vector], 0)
        
        vector_all = vector_all.numpy()
        return vector_all

    def cluster(self, data):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        import numpy as np

        vector_all = self.get_vectors(data)
        # vector_all = np.load('../../data_hw7/simple_all.npy')
        pca = PCA(n_components= 128, whiten= True, random_state= 1)
        vector_all = pca.fit_transform(vector_all)
        # print('Shape: ', vector_all.shape)
        
        clusterer = KMeans(n_clusters= self.cluster_num, random_state= 0).fit(vector_all)
        
        cluster_ids = clusterer.labels_
        celebA = isCelebA()
        cluster_id_consider = cluster_ids[:len(celebA)]
        
        # print(np.bincount(cluster_id_consider[celebA == 0]))
        # print(np.bincount(cluster_id_consider[celebA == 1]))
        celebA_id = [1]
        test_case = get_test_case(self.testcase)
        count = test_case.shape[0]
        same_dataset = []
        for i in range(count):
            cluster_id1 = cluster_ids[test_case[i, 0]]
            cluster_id2 = cluster_ids[test_case[i, 1]]
            id1_celebA = (cluster_id1 in celebA_id)
            id2_celebA = (cluster_id2 in celebA_id)
            if (id1_celebA and  id2_celebA) or ((not id1_celebA) and (not id2_celebA)):
                same_dataset.append(1)
            else:
                same_dataset.append(0)
        self.write_submission(same_dataset)

    def test(self, data):
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import numpy as np
        from matplotlib import pyplot as plt

        vector_origin = np.load('../../data_hw7/simple_all.npy')
        vector_test = np.load('../../data_hw7/vector_vis.npy')
        vector_all = np.vstack([vector_origin, vector_test])
        pca = PCA(n_components= 128, whiten= True, random_state= 1)
        vector_all = pca.fit_transform(vector_all)

        clusterer = KMeans(n_clusters= 2, random_state= 0).fit(vector_all)
        cluster_ids = clusterer.labels_

        # celebA = isCelebA()
        # cluster_id_consider = cluster_ids[:len(celebA)]
        # print(np.bincount(cluster_id_consider[celebA == 0]))
        # print(np.bincount(cluster_id_consider[celebA == 1]))
        celebA_id = 1
        vector_test = vector_test[-5000:]
        label_test  = cluster_ids[-5000:]
        #cluster = PCA(n_components= 2, whiten= True, random_state= 1)
        cluster = TSNE(n_components= 2)
        vector_test = cluster.fit_transform(vector_test)

        plt.title('Prediction')
        plt.scatter(vector_test[label_test != celebA_id][:,0], vector_test[label_test != celebA_id][:, 1], c= 'r')
        plt.scatter(vector_test[label_test == celebA_id][:,0], vector_test[label_test == celebA_id][:, 1], c= 'b')
        plt.savefig('images/visualization_pred.png')
        plt.close()

        plt.title('Ground Truth')
        plt.scatter(vector_test[2500:][:,0], vector_test[2500:][:, 1], c= 'r')
        plt.scatter(vector_test[:2500][:,0], vector_test[:2500][:, 1], c= 'b')
        plt.savefig('images/visualization_gt.png')
        plt.close()

def test():
    import numpy as np
    from sklearn.manifold import TSNE
    x = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    model = TSNE(n_components= 2)
    x_embedded = model.fit_transform(x)
    print(x_embedded)
    
if __name__ == '__main__':
    test()
