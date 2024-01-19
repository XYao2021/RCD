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
import time
from datetime import date
import os


if device != 'cpu':
    current_device = torch.cuda.current_device()
    torch.cuda.set_device(current_device)

if __name__ == "__main__":  #TODO: Why use this sentence
    ACC = []
    LOSS = []
    COMM = []

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
        if DISTRIBUTION == 'Dirichlet':
            client_data = Sample.Synthesize_sampling(alpha=ALPHA)
        elif DISTRIBUTION == 'Single':
            client_data = Sample.DL_sampling_single()
        else:
            raise Exception('This data distribution method has not been embedded')

        client_train_loader = []
        client_residual = []
        client_compressor = []
        Models = []
        client_weights = []
        client_tmp = []
        client_accumulate = []
        # client_est = []

        # max_value = 0.41428038
        # min_value = -0.3107184

        max_value = 0.2782602
        min_value = -0.2472423

        for n in range(CLIENTS):
            model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)
            Models.append(model)
            client_weights.append(model.get_weights())
            client_train_loader.append(DataLoader(client_data[n], batch_size=BATCH_SIZE, shuffle=True))
            client_residual.append(torch.zeros_like(model.get_weights()).to(device))
            # client_tmp.append(model.get_weights())
            # client_accumulate.append(torch.zeros_like(model.get_weights()).to(device))

            client_compressor.append(Top_k(node=n, avg_comm_cost=average_comm_cost, ratio=RATIO))
            # client_compressor.append(Quantization(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value))

        Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network='Random')
        # Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network='Ring')
        check = Check_Matrix(CLIENTS, Transfer.matrix)

        if check != 0:
            raise Exception('The Transfer Matrix Should be Symmetric')
        else:
            print('Transfer Matrix is Symmetric Matrix')

        print(Transfer.neighbors)
        print(Transfer.factor)
        test_model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)

        global_loss = []
        Test_acc = []
        abs_values = []
        abs_X = []
        abs_Bt = []
        Alpha = []
        iter_num = 0
        total_comm_num = 0
        print(Models[0].get_weights(), torch.count_nonzero(Models[0].get_weights()))
        # break

        while True:  # TODO: What is the difference with for loop over clients
            print('SEED ', '|', seed, '|', 'ITERATION ', iter_num)
            Averaged_weights = Transfer.Average(client_weights)
            abs_value = []
            abs_x = []
            abs_bt = []
            alpha = []

            for n in range(CLIENTS):
                Models[n].assign_weights(weights=client_weights[n])
                Models[n].model.train()

                for local_iter in range(ROUND_ITER):
                    images, labels = next(iter(client_train_loader[n]))
                    images, labels = images.to(device), labels.to(device)
                    if data_transform is not None:
                        images = data_transform(images)

                    Models[n].optimizer.zero_grad()
                    pred = Models[n].model(images)
                    loss = Models[n].loss_function(pred, labels)
                    loss.backward()
                    Models[n].optimizer.step()

                Vector_update = Models[n].get_weights()
                Vector_update -= client_weights[n]  # gradient

                Vector_update += Averaged_weights[n]
                Vector_update -= client_weights[n]

                Vector_update, client_residual[n] = client_compressor[n].get_trans_bits_and_residual(iter=iter_num, w_tmp=Vector_update, w_residual=client_residual[n], device=device, channel_quality=None)
                # print(iter_num, n, torch.max(Vector_update), torch.min(Vector_update))
                # abs_x = torch.sum(torch.square(client_weights[n])).item()
                # abs_x.append(torch.sum(torch.square(client_weights[n])).item())
                # abs_value = torch.sum(torch.square(client_residual[n])).item()
                # abs_value.append(torch.sum(torch.square(client_residual[n])).item())
                # abs_bt.append(torch.sum(torch.square(b_t)).item())
                # alpha.append(torch.sum(torch.square(b_t - Vector_update)).item() / torch.sum(torch.square(b_t)).item())

                client_weights[n] += Vector_update

            # abs_values.append(sum(abs_value)/len(abs_value))
            # abs_X.append(abs_x)
            # abs_Bt.append(sum(abs_bt)/len(abs_bt))
            # Alpha.append(sum(alpha)/len(alpha))
            iter_num += 1
            if iter_num % 100 == 0:
                for n in range(CLIENTS):
                    if Models[n].learning_rate > 0:
                        Models[n].learning_rate -= 0.05
                    else:
                        Models[n].learning_rate = 0

            test_weights = [Models[j].get_weights() for j in range(CLIENTS)]
            test_weights = average_weights(test_weights)
            train_loss, train_acc = test_model.accuracy(weights=test_weights, test_loader=train_loader, device=device)
            test_loss, test_acc = test_model.accuracy(weights=test_weights, test_loader=test_loader, device=device)

            global_loss.append(train_loss)
            Test_acc.append(test_acc)
            print('SEED |', seed, '| iteration |', iter_num, '| Global Loss', train_loss, '| Training Accuracy |',
                  train_acc, '| Test Accuracy |', test_acc)

            # if iter_num % 100 == 0:
            #     random.seed(seed)
            #     random.shuffle(client_train_loader[n].dataset)

            if iter_num >= AGGREGATION:
                ACC += Test_acc
                LOSS += global_loss
                COMM.append(total_comm_num)
                break

        del Models
        del client_weights

        torch.cuda.empty_cache()  # Clean the memory cache

    # txt_list = [ACC, '\n', LOSS, '\n', COMP_COST, '\n', COMM_COST]
    # max_value = [np.max(np.array(client_compressor[i].max)) for i in range(len(client_compressor))]
    # min_value = [np.min(np.array(client_compressor[i].min)) for i in range(len(client_compressor))]
    #
    # txt_list = [max_value, '\n', min_value]
    # txt_list = [max_value, '\n', min_value, '\n', abs_Bt, '\n', Alpha]

    txt_list = [ACC, '\n', LOSS]
    # txt_list = [ACC, '\n', LOSS, '\n', Alpha]
    # txt_list = [ACC, '\n', LOSS, '\n', max_value, '\n', min_value, '\n', Alpha]
    f = open('EFD|{}|{}|{}|{}|{}|{}|{}.txt'.format(ROUND_ITER, CLIENTS, NEIGHBORS, RATIO, QUANTIZE_LEVEL, date.today(), time.strftime("%H:%M:%S", time.localtime())), 'w')

    # f = open('PRO_{}_{}.txt'.format(RATIO, ROUND_ITER), 'w')
    for item in txt_list:
        f.write("%s\n" % item)

    # whole length of weights: 39760

    for repeat_time in range(2):
        os.system('say "Program Finished."')
    #  tensor(8.9569, device='cuda:0') tensor(6.7312, device='cuda:0')

    # Residual Compensation Decentralized SGD (RCD-SGD)
