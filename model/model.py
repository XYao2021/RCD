import collections
import os
import sys
import torch
import copy
import collections
from torch.autograd import Variable
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Let path return to the top folder

class Model:
    def __init__(self, random_seed=None, learning_rate=0.1, model_name=None, device=None,
                 flatten_weight=False, pretrained_model_file=None):
        super(Model, self).__init__()  # Equal to super().__init__() in new python version
        if device is None:
            raise Exception('Device is not specified in Model()')
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.model = None
        self.learning_rate = learning_rate
        self.flatten_weight = flatten_weight
        self.key_list = []
        self.shape_list = []
        self.size_list = []

        if model_name == 'FashionMNIST':
            from model.MNISTModel import MNISTModel
            self.model = MNISTModel().to(device)
        elif model_name == 'CIFAR10Model':
            from model.CIFAR10Model import CIFAR10Model, ResNet18
            # self.model = CIFAR10Model().to(device)
            self.model = ResNet18().to(device)

        if pretrained_model_file is not None:
            self.model.load_state_dict(torch.load(pretrained_model_file, map_location=device))

        self.loss_function = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)
        # self.model.to(device)
        self._get_weight_info()

    def _get_weight_info(self):
        weights = self.model.state_dict()
        for key, weight in weights.items():
            if key.split('.')[-1] == 'weight' or key.split('.')[-1] == 'bias':
                shape = list(weight.size())
                self.key_list.append(key)
                self.shape_list.append(shape)
                self.size_list.append(list(weight.reshape((-1, )).size())[0])

    def get_weights(self):
        with torch.no_grad():
            if self.flatten_weight:
                return torch.cat([param.reshape((-1,)) for param in self.model.parameters()])
            else:
                return copy.deepcopy(self.model.state_dict())

    def assign_weights(self, weights):
        if self.flatten_weight:
            self.assign_flatten_weights(weights)
        else:
            self.model.load_state_dict(weights)

    def assign_flatten_weights(self, weights):
        weights_dict = collections.OrderedDict()
        # print(len(weights), sum(self.size_list), self.shape_list, self.size_list, self.key_list)
        new_weights = torch.split(weights, self.size_list)
        for i in range(len(self.shape_list)):
            weights_dict[self.key_list[i]] = new_weights[i].reshape(self.shape_list[i])
        self.model.load_state_dict(weights_dict)

    def accuracy(self, weights, test_loader, device):
        if weights is not None:
            self.assign_weights(weights)
        self.model.eval()
        correct = 0
        loss = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = Variable(images).to(device), Variable(labels).to(device)
                # images, labels = images.to(device), labels.to(device)
                pred = self.model(images)
                loss += self.loss_function(pred, labels).sum()
                pred = pred.data.max(1)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum()
        loss /= len(test_loader.dataset)
        acc = float(correct) / len(test_loader.dataset)
        return loss.item(), acc
