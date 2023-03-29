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
        client_data = Sampling(num_class=len(train_data.classes), num_client=CLIENTS, train_data=train_data, method='uniform', seed=seed).DL_sampling()

        client_train_loader = []
        client_compressor = []
        client_weights = []
        Models = []
        client_residual = []
        client_accumulate = []
        client_tmp = []

        global_loss = []
        Test_acc = []

        Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network='random')
        # Transfer.neighbors = [[0, 3, 6, 9], [1, 4, 7, 8], [2, 3, 5, 6], [0, 2, 3, 9], [1, 4, 7, 8],
        #                       [2, 5, 6, 8], [0, 2, 5, 6], [1, 4, 7, 9], [1, 4, 5, 8], [0, 3, 7, 9]]
        # Transfer.factor = 1 / 4
        test_model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)

        for n in range(CLIENTS):
            model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device,
                          flatten_weight=True, pretrained_model_file=load_model_file)
            Models.append(model)
            client_tmp.append(model.get_weights())
            client_weights.append(model.get_weights())
            client_accumulate.append(torch.zeros_like(model.get_weights()))
            client_residual.append(torch.zeros_like(model.get_weights()))
            client_compressor.append(Normal_compression(node=n, ratio=RATIO))
            client_train_loader.append(DataLoader(client_data[n], batch_size=BATCH_SIZE, shuffle=True))

        iter_num = 0
        while True:  # TODO: What is the difference with for loop over clients
            print('SEED ', '|', seed, '|', 'ITERATION ', iter_num)
            Average_accumulate = Transfer.Average(client_accumulate)
            for j in range(CLIENTS):
                client_weights[j] = client_tmp[j] + CONSENSUS_STEP * Average_accumulate[j]

            for n in range(CLIENTS):
                Vector_update = client_weights[n] - client_accumulate[n]
                Vector_update, _ = client_compressor[n].get_trans_bits_and_residual(w_tmp=Vector_update, w_residual=client_residual[n])
                client_accumulate[n] += Vector_update

                model.assign_weights(weights=client_weights[n])
                model.model.train()

                for i in range(ROUND_ITER):
                    images, labels = next(iter(client_train_loader[n]))
                    images, labels = images.to(device), labels.to(device)
                    if data_transform is not None:
                        images = data_transform(images)

                    Models[n].optimizer.zero_grad()
                    pred = Models[n].model(images)
                    loss = Models[n].loss_function(pred, labels)
                    loss.backward()
                    Models[n].optimizer.step()

                client_tmp[n] = Models[n].get_weights()

            iter_num += 1

            # train_loss, train_acc = test_model.accuracy(weights=client_weights[0], test_loader=train_loader, device=device)
            # test_loss, test_acc = test_model.accuracy(weights=client_weights[0], test_loader=test_loader, device=device)
            test_weights = [Models[i].get_weights() for i in range(CLIENTS)]
            test_weights = average_weights(test_weights)
            train_loss, train_acc = test_model.accuracy(weights=test_weights, test_loader=train_loader, device=device)
            test_loss, test_acc = test_model.accuracy(weights=test_weights, test_loader=test_loader, device=device)
            global_loss.append(train_loss)
            Test_acc.append(test_acc)

            print('SEED |', seed, '| iteration |', iter_num, '| Global Loss', train_loss, '| Training Accuracy |',
                  train_acc, '| Test Accuracy |', test_acc)

            # if iter_num % CHECK == 0:
                # train_loss, train_acc = Models[n].accuracy(weights=client_weights[0], test_loader=train_loader, device=device)
                # test_loss, test_acc = Models[n].accuracy(weights=client_weights[0], test_loader=test_loader, device=device)
                # global_loss.append(train_loss)
                # Test_acc.append(test_acc)

                # print('SEED |', seed, '| iteration |', iter_num, '| Global Loss', train_loss, '| Training Accuracy |', train_acc, '| Test Accuracy |', test_acc)

            if iter_num >= AGGREGATION:
                ACC += Test_acc
                LOSS += global_loss

                break

        del Models
        del client_weights

        torch.cuda.empty_cache()  # Clean the memory cache

    txt_list = [ACC, '\n', LOSS]

    f = open('CHOCO_{}_{}.txt'.format(RATIO, ROUND_ITER), 'w')

    for item in txt_list:
        f.write("%s\n" % item)




