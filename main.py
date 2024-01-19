import torch
import random
import copy
import numpy as np
from torch.utils.data import DataLoader
from model.model import Model
from util.util import *
from compression import *
from partition import *
from config import *
from dataset.dataset import *
from trans_matrix import *
from algorithms.algorithms import *


if device != 'cpu':
    torch.cuda.set_device(device)

if __name__ == "__main__":  #TODO: Why use this sentence
    ACC = []
    LOSS = []
    for seed in Seed_set:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        train_data, test_data = loading(dataset_name=dataset, data_path=dataset_path, device=device)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)

        #TODO: Change the sampling and splitting method to class to accelerate the set-up process
        Sample = Sampling(num_client=CLIENTS, num_class=len(train_data.classes), train_data=train_data, method='uniform', seed=seed)
        client_data = Sample.DL_sampling()
        # print(Sample.target)
        # print(client_data)
        client_train_loader = []
        client_residual = []
        client_compressor = []
        Models = []
        client_weights = []
        client_accumulate = []
        client_tmp = []

        for n in range(CLIENTS):
            model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)
            Models.append(model)
            client_weights.append(model.get_weights())
            client_train_loader.append(DataLoader(client_data[n], batch_size=BATCH_SIZE, shuffle=True))

            client_residual.append(torch.zeros_like(model.get_weights()).to(device))
            client_compressor.append(Normal_compression(node=n, ratio=RATIO))

            client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))
            client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))
            client_tmp.append(model.get_weights())

        Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network='random')
        # Transfer.neighbors = [[0, 3, 6, 9], [1, 4, 7, 8], [2, 3, 5, 6], [0, 2, 3, 9], [1, 4, 7, 8],
        #                       [2, 5, 6, 8], [0, 2, 5, 6], [1, 4, 7, 9], [1, 4, 5, 8], [0, 3, 7, 9]]
        # Transfer.factor = 1/4

        # Transfer.neighbors = [[0, 3, 5], [1, 4, 6], [2, 7, 9], [3, 0, 8], [4, 1, 7], [5, 0, 9], [6, 1, 8], [7, 2, 4], [8, 3, 6], [9, 2, 5]]
        # Transfer.factor = 1/3
        print(Transfer.neighbors)
        print(Transfer.factor)

        train_method = FlexDFL(num_clients=CLIENTS, models=Models, data_transform=data_transform, client_weights=client_weights, data_loader=client_train_loader,
                               compressor=client_compressor, client_residual=client_residual, seed=seed, device=device, transmitter=Transfer, train_test=train_loader, test_test=test_loader)

        iter_num = 0
        global_loss = []
        Test_acc = []
        while True:
            client_weights, client_residual, Models = train_method.training(iter_round=ROUND_ITER, client_weights=client_weights, client_residual=client_residual, models=Models)
            test_weights = [Models[i].get_weights() for i in range(CLIENTS)]
            test_weights = average_weights(test_weights)
            train_loss, train_acc = Models[0].accuracy(weights=test_weights, test_loader=train_loader, device=device)
            test_loss, test_acc = Models[0].accuracy(weights=test_weights, test_loader=test_loader, device=device)
            global_loss.append(train_loss)
            Test_acc.append(test_acc)
            print('SEED |', seed, '| iteration |', iter_num, '| Global Loss', train_loss, '| Training Accuracy |',
                  train_acc, '| Test Accuracy |', test_acc)
            iter_num += 1
