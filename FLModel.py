# Definition of Federated Learning Model (PyTorch version)
import torch
from torch import nn


class LocalFCFModel(nn.Module):
    """
    Local Model of Federated Funk SVD, all private rating record and parameters are stored at client device.
    Each client take their own rating record and user factor vector.
    Item factor vector is shared by all clients, and should NOT be update at local device (client)
    """
    def __init__(self, loc_lr, features, loc_rating, _lambda, device=None):
        super(LocalFCFModel, self).__init__()
        self.device = device
        self.lr = loc_lr
        self.Lambda = _lambda
        self.item_factor = None
        self.rating = loc_rating
        self.mask = self.rating > 0
        self.user_factor = nn.Parameter(nn.init.normal_(torch.empty(1, features), std=0.35), requires_grad=True)
        self.H = 0  # history of gradients (y')

    def forward(self, item_factor):
        """
        Input the server model (item factor vector)
        Return the prediction of ratings with shape [1, # movies]
        """
        return self.user_factor.mm(item_factor)

    def loss_obj(self):
        """Calculate loss of local model"""
        loss = (((self.forward(self.item_factor) - self.rating)*self.mask)**2).sum()
        regularization = self.Lambda * (torch.norm(self.user_factor)**2 +
                                        (torch.norm(self.item_factor, dim=0)*self.mask)**2)
        return loss + regularization.sum()

    def recv_item_factor(self, item_factor):
        """Get the shared item factor"""
        self.item_factor = item_factor.clone().detach().to(self.device)
        self.item_factor.requires_grad_(True)

    def loc_rmse(self):
        return (((self.forward(self.item_factor) - self.rating)*self.mask)**2).sum()

    def get_user_factor(self):
        return self.user_factor.detach().cpu()

    def add_his_grad(self, grad):
        """Should be invoke after finishing local iterations"""
        self.H += grad


class ServerFCFModel(nn.Module):
    """
    Server Model of Federated Funk SVD
    The item factor vector is stored at server and send to all clients.
    The gradients are aggregated after all clients completing local update.
    Item factor vector is updated via this gradient
    """
    def __init__(self, srv_lr, item_num, features, device=None):
        super(ServerFCFModel, self).__init__()
        self.lr = srv_lr
        self.device = device
        self.features = features
        self.item_factor = nn.init.normal_(torch.empty(self.features, item_num), std=0.35).to(self.device)

    def update(self, grad):
        """
        Update the shard item factor vector
        """
        grad = grad.to(self.device)
        self.item_factor = self.item_factor - self.lr * grad
        #return self.item_factor.clone().detach()

    def get_item_factor(self):
        return self.item_factor.detach()

    def forward(self):
        pass