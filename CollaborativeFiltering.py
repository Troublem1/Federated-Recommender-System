#! /home/jyfan/venv/bin/python
"""
Collaborative Filtering Recommend System (Serial version), implement in TensorFlow
"""
import random
import numpy as np
import pandas as pd
import torch
from torch import nn


class CFModel(nn.Module):
    """Collaborative Filtering Model in Pytorch"""
    def __init__(self, users, movies, features, device):
        super(CFModel, self).__init__()
        self.device = device
        self.NUM_USER = users
        self.NUM_MOVIE = movies
        self.features = features
        self.params = nn.ParameterDict({
            'X': nn.Parameter(nn.init.normal_(torch.empty(self.NUM_USER, self.features), std=0.35), requires_grad=True),
            'Y': nn.Parameter(nn.init.normal_(torch.empty(self.features, self.NUM_MOVIE), std=0.35), requires_grad=True)
        })

    def forward(self):
        return self.params['X'].mm(self.params['Y'])


class CollaborativeFiltering(object):
    def __init__(self, data_path):
        self.NUM_USER = None
        self.NUM_MOVIE = None

        self.rating = None  # rating record, rating[i] represent for the rating record of user $i$
        self.model = None  # user(X)&item(Y) factor vector, shape(X)=(#users, #features), shape(Y)=(#features, #movies)
        self.mask = None  # mask matrix, mask = (rating > 1)
        self.test_case = None  # test case (20% of total rating data)

        self.feature = 5  # feature is 10 by default
        self.Lambda = 0.02  # penalty factor
        self.lr = 1e-4  # learning rate

        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        self.load_data(data_path)
        self.init_model()

    def init_model(self):
        """
        Initialize Models.
        The elements of user/item factor vector are generated with a normal distribution.
        User factor vector (X): 2D - matrix, shape = (#users, #features)
        Item factor vector (Y): 2D - matrix, shape = (#features, #items)
        R ~ X*Y, where R is the rating matrix (dataset)
        """
        self.model = CFModel(self.NUM_USER, self.NUM_MOVIE, self.feature, self.device)
        self.model.to(self.device)

    def loss_obj(self):
        """
        Calculate the loss value
        """
        loss = (((self.model.forward() - self.rating)*self.mask)**2).sum()

        # calculate regularization
        regularization = 0.0
        p = torch.norm(self.model.params['X'], dim=1)**2
        q = torch.norm(self.model.params['Y'], dim=0)**2
        regularization = (p*self.mask.transpose(0, 1)).sum() + (q*self.mask).sum()
        return loss + self.Lambda*regularization

    def RMSE(self):
        """
        Calculate root mean squared error of test data
        """
        predict = self.model.forward().cpu().detach().numpy()
        loss = 0.0
        for (uid, item, target) in self.test_case:
            loss += (predict[uid][item] - target)**2
        loss /= float(len(self.test_case))
        return np.sqrt(loss)

    def train(self):
        """
        Update parameters (user&item factor vector) via ``Alternating least squares techniques"
        """
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        optimizer.zero_grad()
        loss = self.loss_obj()
        loss.backward()
        #print("y=", self.model.params['Y'].grad)
        optimizer.step()

    def save_model(self, rnd):
        """Save the model"""
        np.save("CF_item_" + str(rnd) + "_rnd.npy", self.model.params['Y'].cpu().deatch().numpy())
        np.save("CF_user_" + str(rnd) + "_rnd.npy", self.model.params['X'].cpu().deatch().numpy())

    def set_lambda(self, _lambda):
        self.Lambda = _lambda

    def load_data(self, data_path):
        """
        Generate federated learning data
        :param data_path:
        :return: user-item matrix (2D - numpy array, shape=(# users, # movies))
        """
        data_file = "1m_rating.npy"
        try:
            rating = np.load(data_file)
            self.NUM_USER = rating.shape[0]
            self.NUM_MOVIE = rating.shape[1]
        except FileNotFoundError:
            print("FileNotFound, construct rating matrix.")
            ratings_df = pd.read_csv(data_path + 'ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'Time'])
            movies_df = pd.read_csv(data_path + 'movies.dat', sep='::', names=['movieId', 'name', 'type'])
            self.NUM_USER = ratings_df['userId'].drop_duplicates().size
            self.NUM_MOVIE = movies_df.index.size

            movie_id_mapping = {}
            id_count = 1
            for i in movies_df['movieId']:
                movie_id_mapping[int(i)] = int(id_count)
                id_count += 1

            rating = np.zeros((self.NUM_USER, self.NUM_MOVIE), dtype=np.float32)
            for index, row in ratings_df.iterrows():
                row_id = int(row['userId']) - 1
                col_id = movie_id_mapping[int(row['movieId'])] - 1
                rating[row_id][col_id] = row['rating']
            np.save(data_file, rating)
        print("# user", self.NUM_USER)
        print("# movie", self.NUM_MOVIE)
        rating, self.test_case = self.split_dataset(rating)
        self.rating = torch.tensor(rating).to(self.device)
        self.mask = (self.rating > 0)*1.0
        self.mask.to(self.device)
        print("test data:", len(self.test_case))
        print("train data:", self.mask.cpu().numpy().sum())

    def split_dataset(self, rating):
        """
        Split train set and test set. For each user, 20% rating records (at least 4) are selected as test case
        """
        rating_num = ((rating > 0)*1.0).sum(1)
        test_case = []
        for uid in range(self.NUM_USER):
            index = np.where(rating[uid] > 0.1)
            index = list(index[0])
            test_sample = random.sample(index, int(0.2 * rating_num[uid]))
            for i in test_sample:
                test_case.append((uid, i, rating[uid][i]))
                rating[uid][i] = 0
        return rating, test_case
