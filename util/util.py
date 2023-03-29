import numpy as np
import random
import copy
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

def sampler(dataset, num_nodes):
    pass

def MINIST_sample(num_classes, num_clients):  # For number of clients greater or equal to number of classes
    org_set = np.arange(num_classes, dtype=int)
    samples = np.array([], dtype=int)
    while num_clients > 0:
        np.random.shuffle(org_set)
        if num_clients <= num_classes:
            sample = org_set[:num_clients]
        else:
            sample = org_set
        samples = np.append(samples, sample, axis=0)
        num_clients -= num_classes
    return samples

def split_data(sample, train_data):
    data = [[] for _ in range(len(sample))]
    for i in range(len(train_data.targets)):
        for j in range(len(sample)):
            if train_data.targets[i] == sample[j]:
                data[j].append(train_data[i])
    return data

class Sampling:
    def __init__(self, num_class, num_client, train_data, method, seed):
        super().__init__()
        self.num_class = num_class
        self.num_client = num_client
        self.train_data = train_data
        self.method = method
        self.seed = seed

        self.partition = None
        self.set_length = None
        self.sample_data = [[] for _ in range(num_class)]
        self.classes = np.arange(self.num_class)
        self.target = None

        self._initialize()

    def _initialize(self):
        if self.method == 'uniform':
            self.partition = np.ones(self.num_client) / self.num_client
        elif self.method == 'random':  # TODO: Need to consider the relationship with BATCH_SIZE
            self.partition = np.random.random(self.num_client)
            self.partition /= np.sum(self.partition)

        for i in range(len(self.train_data)):
            for j in range(self.num_class):
                # print(len(self.sample_data[j]))
                if self.train_data.targets[i] == j:
                    self.sample_data[j].append(self.train_data[i])
        self.set_length = len(self.sample_data[0])

    # def DL_sampling(self):
    #     Sampled_data = [[] for _ in range(self.num_client)]
    #     num_samples = int(len(self.train_data)/self.num_client)
    #     samples = np.arange(len(self.train_data), dtype=int)
    #     samples = np.split(samples, self.num_client)
    #     # print(samples)
    #     for i in range(self.num_client):
    #         for j in range(num_samples):
    #             Sampled_data[i].append(self.train_data[samples[i][j]])
    #     return Sampled_data
    #
    # def DL_sampling(self):
    #     np.random.seed(seed=self.seed)
    #     Sampled_data = [[] for _ in range(self.num_client)]
    #     num_samples = (self.set_length * self.partition).astype(int)
    #     for i in range(self.num_client):
    #         for j in range(self.num_class):
    #             np.random.shuffle(self.sample_data[j])
    #             Sampled_data[i] += self.sample_data[j][num_samples[i] * i:num_samples[i] * (i+1)]
    #     # np.random.shuffle(Sampled_data)
    #     return Sampled_data

    def DL_sampling(self):
        Sampled_data = []
        target = np.arange(self.num_client)
        target %= self.num_class
        np.random.shuffle(target)
        self.target = target
        for i in range(self.num_client):
            Sampled_data.append(self.sample_data[target[i]])
        return Sampled_data

    # def DL_sampling(self):
    #     Sampled_data = [[] for _ in range(self.num_client)]
    #     sample = np.arange(self.num_client, dtype=int)
    #     sample %= self.num_class
    #     self.target = [[] for _ in range(self.num_client)]
    #     np.random.seed(self.seed)
    #     np.random.shuffle(sample)
    #     sample_1 = copy.deepcopy(sample)
    #     np.random.shuffle(sample_1)
    #     for i in range(self.num_client):
    #         self.target[i].append(sample[i])
    #         self.target[i].append(sample_1[i])
    #     for n in range(self.num_client):
    #         for j in range(len(self.train_data)):
    #             if self.train_data.targets[j] in self.target[n]:
    #                 Sampled_data[n].append(self.train_data[j])
    #         np.random.shuffle(Sampled_data[n])
    #     return Sampled_data

    # def CL_sampling(self):  #TODO: Not complete
    #     num_clients = self.num_client
    #     np.random.seed(self.seed)
    #
    #     org_set = np.arange(self.num_class, dtype=int)
    #     samples = np.array([], dtype=int)
    #     while num_clients > 0:
    #         np.random.shuffle(self.classes)
    #         if num_clients <= self.num_class:
    #             sample = self.classes[:num_clients]
    #         else:
    #             sample = self.classes
    #         samples = np.append(samples, sample, axis=0)
    #         num_clients -= self.num_class
    #
    #     data = [[] for _ in range(self.num_client)]
    #     for i in range(len(self.train_data)):
    #         for j in range(self.num_client):
    #             if self.train_data.targets[i] == samples[j]:
    #                 data[j].append(self.train_data[i])
    #     return data

def average_weights(weights):
    for i in range(len(weights)):
        if i == 0:
            total = weights[i]
        else:
            total += weights[i]
    total = torch.div(total, torch.tensor(len(weights)))
    return total
