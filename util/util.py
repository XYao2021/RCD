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
        self.set_length = int(len(self.train_data)/self.num_class)
        self.sample_data = [[] for _ in range(num_class)]
        self.classes = np.arange(self.num_class)
        self.target = None
        self.dataset = []

        self._initialize()

    def _initialize(self):
        if self.method == 'uniform':
            self.partition = np.ones(self.num_client) / self.num_client
        elif self.method == 'random':  # TODO: Need to consider the relationship with BATCH_SIZE
            self.partition = np.random.random(self.num_client)
            self.partition /= np.sum(self.partition)
        for i in range(self.num_class):
            tmp_data = []
            for j in range(len(self.train_data)):
                if self.train_data.targets[j] == self.classes[i]:
                    tmp_data.append(self.train_data[j])
            self.dataset.append(tmp_data)

    def DL_sampling_single(self):
        np.random.seed(self.seed)
        Sampled_data = [[] for _ in range(self.num_client)]
        self.target = np.arange(self.num_client)
        self.target %= self.num_class
        np.random.shuffle(self.target)
        for i in range(self.num_client):
            for j in range(len(self.train_data)):
                if self.train_data.targets[j] == self.target[i]:
                    Sampled_data[i].append(self.train_data[j])
        return Sampled_data

    def Complete_Random(self):
        np.random.seed(self.seed)
        Sampled_data = [[] for _ in range(self.num_client)]
        indices = np.arange(len(self.train_data), dtype=int)
        np.random.shuffle(indices)
        k = int(len(self.train_data)/self.num_client)
        for i in range(self.num_client):
            client_indices = indices[i*k: (i+1)*k]
            for index in client_indices:
                Sampled_data[i].append(self.train_data[index])
        return Sampled_data

    def Synthesize_sampling(self, alpha):
        Alpha = [alpha for i in range(self.num_class)]
        # Generate samples from the Dirichlet distribution
        samples = np.random.dirichlet(Alpha, size=self.num_client)
        # Print the generated samples
        num_samples = []
        for sample in samples:
            sample = np.array(sample) * len(self.dataset[0])
            num_samples.append([int(round(i, 0)) for i in sample])
        Sample_data = [[] for i in range(self.num_client)]
        for client in range(self.num_client):
            for i in range(self.num_class):
                class_samples = num_samples[client][i]
                tmp_data = random.sample(self.dataset[i], k=class_samples)
                # tmp_data_1 = random.sample(self.dataset[i], k=class_samples)
                Sample_data[client] += tmp_data
                # Sample_data[client] += tmp_data_1
            np.random.shuffle(Sample_data[client])
        return Sample_data

def average_weights(weights):
    for i in range(len(weights)):
        if i == 0:
            total = weights[i]
        else:
            total += weights[i]
    total = torch.div(total, torch.tensor(len(weights)))
    return total

def Check_Matrix(client, matrix):
    count = 0
    for i in range(client):
        if np.count_nonzero(matrix[i] - matrix.transpose()[i]) == 0:
            pass
        else:
            count += 1
    return count
