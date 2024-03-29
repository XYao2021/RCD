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


if device != 'cpu':
    current_device = torch.cuda.current_device()
    torch.cuda.set_device(current_device)

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
        if DISTRIBUTION == 'Dirichlet':
            if ALPHA == 0:
                client_data = Sample.DL_sampling_single()
            elif ALPHA > 0:
                client_data = Sample.Synthesize_sampling(alpha=ALPHA)
        elif DISTRIBUTION == 'Single':
            client_data = Sample.DL_sampling_single()
        else:
            raise Exception('This data distribution method has not been embedded')

        client_train_loader = []
        client_compressor = []
        client_weights = []
        Models = []
        client_residual = []
        client_accumulate = []
        client_tmp = []
        client_partition = []

        global_loss = []
        Test_acc = []

        Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network='Ring')
        # Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network='Random')
        test_model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)
        print(Transfer.neighbors)
        print(Transfer.factor)

        # check = Check_Matrix(CLIENTS, Transfer.matrix)
        # if check != 0:
        #     raise Exception('The Transfer Matrix Should be Symmetric')
        # else:
        #     print('Transfer Matrix is Symmetric Matrix')
        # max_value = 0.29974118
        # min_value = -0.30779636

        max_value = 0.35543507
        min_value = -0.30671167

        for n in range(CLIENTS):
            model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device,
                          flatten_weight=True, pretrained_model_file=load_model_file)
            Models.append(model)
            client_tmp.append(model.get_weights())
            client_weights.append(model.get_weights())
            client_accumulate.append(torch.zeros_like(model.get_weights()))
            client_residual.append(torch.zeros_like(model.get_weights()))

            if METHOD == 'Lyapunov':
                # client_compressor.append(Lyapunov_compression_1(node=n, avg_comm_cost=average_comm_cost, V=V, W=W))
                client_compressor.append(Lyapunov_compression_Q(node=n, avg_comm_cost=average_comm_cost, V=V, W=W, max_value=max_value, min_value=min_value))
                client_partition.append(Lyapunov_Participation(node=n, average_comp_cost=average_comp_cost, V=V, W=W, seed=seed))
            elif METHOD == 'Fixed':
                client_compressor.append(Fixed_Compression(node=n, avg_comm_cost=average_comm_cost, ratio=RATIO))
                client_partition.append(Fixed_Participation(average_comp_cost=average_comp_cost))

            client_train_loader.append(DataLoader(client_data[n], batch_size=BATCH_SIZE, shuffle=True))
            # participation = np.random.choice(np.arange(AGGREGATION), 583)
            # client_partition.append(participation)
            # client_partition.append(Fixed_Participation(average_comp_cost=average_comp_cost))

        iter_num = 0
        update_times = 0
        # CHOCO Algorithm
        while True:  # TODO: What is the difference with for loop over clients
            print('SEED ', '|', seed, '|', 'ITERATION ', iter_num)
            Vector_Update = []
            Averaged_weights = Transfer.Average(client_weights)

            for n in range(CLIENTS):
                Models[n].assign_weights(weights=client_weights[n])
                Models[n].model.train()

                # if iter_num in client_partition[n]:
                qt = client_partition[n].get_q(iter_num)
                if np.random.binomial(1, qt) == 1:
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
                    Vector_update -= client_weights[n]
                    Vector_update += Averaged_weights[n]

                else:
                    Vector_update = Averaged_weights[n]

                # gamma_t = [0]
                Vector_update -= client_weights[n]
                Vector_update, _ = client_compressor[n].get_trans_bits_and_residual(w_tmp=Vector_update, w_residual=client_residual[n], iter=iter_num, device=device, channel_quality=None)  # Not work?

                client_weights[n] += Vector_update

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

            if iter_num >= AGGREGATION:
                ACC += Test_acc
                LOSS += global_loss

                break

        del Models
        del client_weights

        torch.cuda.empty_cache()  # Clean the memory cache

    txt_list = [ACC, '\n', LOSS]

    f = open('DCD|{}|{}|{}|{}.txt'.format(RATIO, ROUND_ITER, AGGREGATION, time.strftime("%H:%M:%S", time.localtime())), 'w')

    for item in txt_list:
        f.write("%s\n" % item)
