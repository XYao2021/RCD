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
        client_est = []

        global_loss = []
        Test_acc = []

        TMP_LEARNING_RATE = LEARNING_RATE
        Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network='Ring')
        # Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network='random')

        test_model = Model(random_seed=seed, learning_rate=TMP_LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)
        print(Transfer.neighbors)
        print(Transfer.factor)
        #
        # check = Check_Matrix(CLIENTS, Transfer.matrix)
        # if check != 0:
        #     raise Exception('The Transfer Matrix Should be Symmetric')
        # else:
        #     print('Transfer Matrix is Symmetric Matrix')
        "using mean value of all max/min values"
        # max_value = 3.6763039586037496
        # min_value = -3.4725098398225
        "using mean value of top-20 max/min values"
        # max_value = 17.6002401753125
        # min_value = -16.843656876875002

        # max_value = 34.90439
        # min_value = -24.813738

        # max_value = 94.066895
        # min_value = -95.25409

        max_value = 66.03526
        min_value = -57.940025

        for n in range(CLIENTS):
            model = Model(random_seed=seed, learning_rate=TMP_LEARNING_RATE, model_name=model_name, device=device,
                          flatten_weight=True, pretrained_model_file=load_model_file)
            Models.append(model)
            client_tmp.append(model.get_weights())
            client_weights.append(model.get_weights())
            client_est.append(model.get_weights())

            client_residual.append(torch.zeros_like(model.get_weights()))

            # client_compressor.append(Top_k(node=n, avg_comm_cost=average_comm_cost, ratio=RATIO))
            client_compressor.append(Quantization(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value))

            client_train_loader.append(DataLoader(client_data[n], batch_size=BATCH_SIZE, shuffle=True))
            # client_partition.append(Fixed_Participation(average_comp_cost=average_comp_cost))

        iter_num = 1
        update_times = 0
        # CHOCO Algorithm
        while True:  # TODO: What is the difference with for loop over clients
            print('SEED ', '|', seed, '|', 'ITERATION ', iter_num)
            Averaged_weights = Transfer.Average(client_est)

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
                Vector_update -= client_weights[n]
                Vector_update += Averaged_weights[n]

                z_vector = (1 - 0.5 * iter_num) * client_weights[n] + 0.5 * iter_num * Vector_update
                z_vector, _ = client_compressor[n].get_trans_bits_and_residual(w_tmp=z_vector, w_residual=client_residual[n], iter=iter_num, device=device, channel_quality=None)  # Not work?

                client_est[n] = (1 - 2/iter_num) * client_est[n] + 2/iter_num * z_vector
                client_weights[n] = Vector_update

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

            if iter_num >= AGGREGATION + 1:
                ACC += Test_acc
                LOSS += global_loss

                break

        del Models
        del client_weights

        torch.cuda.empty_cache()  # Clean the memory cache

    # max_value = [np.max(np.array(client_compressor[i].max)) for i in range(len(client_compressor))]
    # min_value = [np.min(np.array(client_compressor[i].min)) for i in range(len(client_compressor))]
    #
    # txt_list = [max_value, '\n', min_value]
    #
    txt_list = [ACC, '\n', LOSS]

    f = open('ECD|{}|{}|{}|{}.txt'.format(RATIO, ROUND_ITER, AGGREGATION, time.strftime("%H:%M:%S", time.localtime())), 'w')

    for item in txt_list:
        f.write("%s\n" % item)

    # for repeat_time in range(2):
    #     os.system('say "Program Finished."')
