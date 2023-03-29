import argparse
import torch
import os
import torchvision.transforms as transforms
from torchvision import datasets


parse = argparse.ArgumentParser()
parse.add_argument('-data', type=str, default='fashion', help='choose target dataset')  # Like CIFAR10, MNIST, FashionMNIST, CIFAR100
parse.add_argument('-pretrained-model', type=str, default='', help='pretrain model path')
parse.add_argument('-method', type=str, default='Lyapunov', help='model method')

parse.add_argument('-agg', type=int, default=1000, help='Global Aggregation times/ total iterations')
parse.add_argument('-iter_round', type=int, default=1, help='Local Training Times: Iterations')
parse.add_argument('-acc_point', type=int, default=20, help='Accuracy check point')

parse.add_argument('-lr', type=float, default=0.1, help='Learning Rate of the Model')
parse.add_argument('-bs', type=int, default=32, help='Batch Size for model')
parse.add_argument('-bs_test', type=int, default=128, help='Batch Size for test model')
parse.add_argument('-cn', type=int, default=10, help='Client Number')
parse.add_argument('-nn', type=int, default=10, help='Number of Neighbors')

parse.add_argument('-V', type=float, default=0.02, help='Lyapunov V value, constant weight to ensure the average of p(t) close to optimal ')
parse.add_argument('-W', type=float, default=1.0, help='Lyapunov W value, initial queue length')
parse.add_argument('-avg_comm', type=float, default=0.01, help='Average communication cost')
parse.add_argument('-avg_comp', type=float, default=0.25, help='Average computation cost')

parse.add_argument('-seed', type=int, default=11, help='random seed for pseudo random model initial weights')
parse.add_argument('-ns', type=int, default=2, help='Number of seeds for simulation')
parse.add_argument('-ratio', type=float, default=0.01, help='the ratio of non-zero elements that the baseline want to transfer')

parse.add_argument('-consensus', type=float, default=0.01, help='Consensus step for CHOCO')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = parse.parse_args()
print(', '.join(f'{k}={v}' for k, v in vars(args).items()))

average_comm_cost = args.avg_comm
average_comp_cost = args.avg_comp
V = args.V  # Lyapunov V value
W = args.W  # Lyapunov initial queue length W
LEARNING_RATE = args.lr
RATIO = args.ratio
CONSENSUS_STEP = args.consensus

dataset_path = os.path.join(os.path.dirname(__file__), 'data')

if args.data == 'fashion':
    model_name = 'FashionMNIST'
    dataset = 'FashionMNIST'
#     train_data = datasets.FashionMNIST(root='data', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
#     test_data = datasets.FashionMNIST(root='data', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
elif args.data == 'CIFAR10':
    model_name = 'CIFAR10Model'
    dataset = 'CIFAR10'
#     train_data = datasets.CIFAR10(root='data', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
#     test_data = datasets.CIFAR10(datasets.CIFAR10(root='data', train=False, transform=transforms.ToTensor(), target_transform=None, download=True))
else:
    raise Exception('Unknown dataset, need to update')

AGGREGATION = args.agg  # This time aggregation equal to total iteration for 1-SGD
BATCH_SIZE = args.bs
BATCH_SIZE_TEST = args.bs_test
if args.pretrained_model != '':
    load_model_file = args.pretrained_model
else:
    load_model_file = None

if dataset == 'CIFAR10' or dataset == 'CIFAR100':
    data_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip()])
else:
    data_transform = None

Seed_number = args.ns
# if Seed_number < 10:
#     raise Exception('At least 10 different seeds for test')
Seed_up = args.seed

Seed_set = [i for i in range(Seed_up-Seed_number, Seed_up)]
CLIENTS = args.cn
NEIGHBORS = args.nn
METHOD = args.method
ROUND_ITER = args.iter_round
CHECK = args.acc_point
# print('SEED SET: ', Seed_set)
