# Federated Collaborative Filtering.
# Serial version, implement in PyTorch.
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity

from FLModel import LocalFCFModel, ServerFCFModel


class FederatedCF(object):
    def __init__(self, data_path, attacker=0, defense_alg=False):
        self.client_num = None
        self.NUM_USER = None
        self.NUM_MOVIE = None
        self.E = 1      # each client perform E-round iteration between each communication round
        self.attacker = attacker            # number of attackers (malicious user)
        self.defense_alg = defense_alg      # whether use defense algorithm to avoid attackers
        self.alpha = []                     # re-scale lr of each clients (``FoolsGold" Alg.)
        self.cs = None                      # St-weighted cosine similarity for ``FoolsGold" Alg.

        self.clients = None         # a list of client models
        self.server_model = None    # the server model

        self.test_case = None       # test case (20% of total rating data)
        self.rating = None          # global rating record, rating[i] represent for the rating record of user $i$
        self.user_factor = None     # local factor vector, shape=(#users, #features)
        self.item_factor = None     # shared factor vector, shape=(#features, #movies)
        self.mask = None            # mask matrix, mask = (rating > 1)

        self.feature = 5    # feature is 50 by default
        self.Lambda = 0.02  # penalty factor

        self.client_lr = None  # Python List with NUM_USER lr for each client
        self.server_lr = 1e-2

        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        self.load_data(data_path)
        self.init_model()
        print("# user", self.NUM_USER)
        print("# client", self.client_num)
        print("# item", self.NUM_MOVIE)
        print("size", self.mask.shape)

    def init_model(self):
        """
        Initialize model (local user factor vectors and a shared item factor vector)
        Generate a shared item factor vectors, the user factor vectors should be generated independently
        """
        self.clients = [LocalFCFModel(self.client_lr[uid], self.feature,
                                      self.rating[uid], self.Lambda, self.device).to(self.device)
                        for uid in range(self.client_num)]
        self.server_model = ServerFCFModel(self.server_lr, self.NUM_MOVIE, self.feature, self.device).to(self.device)
        self.broad_cast()

    def client_update(self, uid):
        """
        Update user factor vectors user_factor[uid] for a client
        Return gradients of item factor vector to server
        """
        optimizer = torch.optim.SGD(self.clients[uid].parameters(), lr=self.clients[uid].lr)
        for i in range(self.E):
            optimizer.zero_grad()
            loss = self.clients[uid].loss_obj()
            loss.backward()
            optimizer.step()
        return self.clients[uid].item_factor.grad.clone().detach()

    def global_update(self):
        """Perform one round of global update"""
        grads = []
        for uid in range(self.client_num):
            grads.append(self.client_update(uid))
            self.clients[uid].add_his_grad(grads[-1])
            torch.cuda.empty_cache()
        # update the alpha
        if self.defense_alg:
            self.FoolsGold()
            for i in range(len(grads)):
                grads[i] *= self.alpha[i]
        grads = sum(grads) / self.client_num
        self.server_model.update(grads)
        self.broad_cast()

    def broad_cast(self):
        """Broadcast new item factor to all clients"""
        item_factor = self.server_model.get_item_factor()
        for uid in range(self.client_num):
            self.clients[uid].recv_item_factor(item_factor)

    def global_loss(self):
        """Calculate global loss"""
        loss = 0.0
        for uid in range(self.client_num):
            loss += self.clients[uid].loss_obj()
        return loss

    def RMSE(self):
        """
        Calculate the RMSE of current model of test set
        """
        predict = [self.clients[uid].forward(self.server_model.get_item_factor()).cpu().detach()
                   for uid in range(self.client_num-self.attacker)]
        loss = 0.0
        for (uid, item, target) in self.test_case:
            loss += (predict[uid][0][item] - target)**2
        loss /= float(len(self.test_case))
        return np.sqrt(loss)

    def set_loc_iter_round(self, loc_round):
        self.E = loc_round

    def set_penalty_factor(self, _lambda):
        self.Lambda = _lambda

    def set_lr(self, server_lr, client_lr=None):
        if server_lr:
            self.server_lr = server_lr
            self.server_model.lr = self.server_lr
        if client_lr:
            self.client_lr = [client_lr for i in range(self.client_num)]
            for uid in range(self.client_num):
                self.clients[uid].lr = client_lr

    def save_global_model(self, _round):
        """Return the global model -- user and item matrix"""
        global_user_factor = torch.concat(self.user_factor, 0)
        np.save("FCF_item_" + str(_round) + "_rnd.npy", self.item_factor)
        np.save("FCF_user_" + str(_round) + "_rnd.npy", global_user_factor)

    def load_data(self, data_path):
        """
        Generate federated learning data
        Return user-item matrix (2D - numpy array, shape=(# users, # movies))
        """
        #data_file = "rating.npy"
        data_file = "1m_rating.npy"
        """ Construct Rating Matrix """
        try:
            rating = np.load(data_file)
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

        # Add attackers
        if self.attacker > 0:
            rating = self.add_attacker(self.attacker, rating)

        self.NUM_USER = rating.shape[0]
        self.NUM_MOVIE = rating.shape[1]

        rating, self.test_case = self.split_dataset(rating)

        self.client_num = self.NUM_USER
        self.rating = torch.tensor(rating).to(self.device)
        self.mask = (self.rating > 0)*1.0
        self.mask.to(self.device)

        self.client_lr = [1e-4 for i in range(self.NUM_USER)]
        print("test data:", len(self.test_case))
        print("train data:", self.mask.cpu().numpy().sum())

    def split_dataset(self, rating):
        """
        Split train set and test set. For each user, 20% rating records (at least 4) are selected as test case
        """
        rating_num = ((rating > 0) * 1.0).sum(1)
        test_case = []
        for uid in range((self.NUM_USER - self.attacker)):
            # only use those real users to test the rmse score
            index = np.where(rating[uid] > 0.1)
            index = list(index[0])
            test_sample = random.sample(index, int(0.2 * rating_num[uid]))
            for i in test_sample:
                test_case.append((uid, i, rating[uid][i]))
                rating[uid][i] = 0
        return rating, test_case

    def add_attacker(self, attacker_num, rating):
        """Add attacker into the FL system"""
        items = rating.shape[1]
        total_record = (rating > 0.1).sum()
        avg_record = total_record / rating.shape[0]
        single_attack = np.zeros((1, items))

        # Generate attack data randomly
        indexs = random.sample(range(items), int(avg_record))
        for i in indexs:
            single_attack[0][i] = random.randint(1, 5)
        for _ in range(attacker_num):
            rating = np.row_stack((rating, single_attack))
        return rating

    def FoolsGold(self):
        """
        FoolsGold Algorithm
        Calculate similarity of gradients of clients to defense sybils attack
        The attacker's gradients will be penalized
        """
        # flatten gradients (matrix) into vectors
        H = [self.clients[uid].H.cpu().numpy().flatten() for uid in range(self.client_num)]

        # calculate CS_ij
        H = np.array(H)
        cs = cosine_similarity(H)
        cs -= np.eye(self.client_num)
        maxcs = np.max(cs, axis=1)
        for i in range(self.client_num):
            for j in range(self.client_num):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        alpha = 1 - (np.max(cs, axis=1))
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0
        alpha = alpha / np.max(alpha)
        alpha[(alpha == 1)] = .99999

        # Logit function
        alpha = (np.log(alpha / (1 - alpha)) + 0.5)
        alpha[(np.isinf(alpha) + alpha > 1)] = 1
        alpha[(alpha < 0)] = 0
        self.alpha = alpha.copy()
