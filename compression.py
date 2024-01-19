import copy
from abc import ABC
import math
import torch
import numpy as np
import abc
from config import *
from numpy.random import RandomState, SeedSequence
from numpy.random import MT19937
from sklearn.cluster import KMeans


# def communication_cost(node, iter, full_size, trans_size):
#     if trans_size == 0:
#         return 0.0
#     else:
#         rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))
#         rs1 = RandomState(MT19937(SeedSequence(iter * 800 + node + 23457)))
#
#         constant = 0.05  # beta
#         SNR_0 = rs.chisquare(df=2)
#         SNR_1 = rs1.chisquare(df=2)
#
#         if SNR_0 > SNR_1:
#             gamma = 1 / np.log2(1 + SNR_1)
#         elif SNR_1 > SNR_0:
#             gamma = 1 / np.log2(1 + SNR_0)
#         return constant + float(trans_size)/full_size * gamma
#
# def communication_cost_quan(node, iter, full_size, trans_size):
#     if trans_size == 0:
#         return 0.0
#     else:
#         rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))
#         rs1 = RandomState(MT19937(SeedSequence(iter * 800 + node + 23457)))
#
#         constant = 16  # beta
#         SNR_0 = rs.chisquare(df=2)
#         SNR_1 = rs1.chisquare(df=2)
#
#         if SNR_0 > SNR_1:
#             gamma = 1 / np.log2(1 + SNR_1)
#         elif SNR_1 > SNR_0:
#             gamma = 1 / np.log2(1 + SNR_0)
#
#         return constant + trans_size * gamma
#
def communication_cost(node, iter, full_size, trans_size):
    if trans_size == 0:
        return 0.0
    else:
        rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))

        constant = 0.05  # beta
        SNR_0 = rs.chisquare(df=2)

        gamma = 1 / np.log2(1 + SNR_0)
        return constant + (float(trans_size)/full_size) * gamma

def communication_cost_quan(node, iter, full_size, trans_size):
    if trans_size == 0:
        return 0.0
    else:
        rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))

        constant = 8 / full_size  # beta
        SNR_0 = rs.chisquare(df=2)

        gamma = 1 / np.log2(1 + SNR_0)

        return constant + (trans_size/full_size) * gamma

def communication_cost_multiple(node, iter, full_size, trans_size, channel_quality):
    if trans_size == 0:
        return 0.0
    else:
        constant = 0.05  # beta
        # print('neighbors', channel_quality)
        if channel_quality is None:
            rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))
            SNR_0 = rs.chisquare(df=2)
            gamma = 1 / np.log2(1 + SNR_0)
        else:
            neighbors, add_trans = channel_quality
            neighbors = np.setdiff1d(neighbors, node)
            SNR = []
            Gamma_neighbor = []
            for i in range(len(neighbors)):
                rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456 + neighbors[i])))
                SNR.append(rs.chisquare(df=2))
                gamma_neighbor = []
                for k in range(len(add_trans[i])):
                    rs1 = RandomState(MT19937(SeedSequence(iter * 800 + neighbors[i] + 23456 + add_trans[i][k])))
                    gamma_neighbor.append(1 / np.log2(1 + rs1.chisquare(df=2)))
                Gamma_neighbor.append(sum(gamma_neighbor))
            SNR = min(SNR)
            gamma = 1 / np.log2(1 + SNR)
            gamma += sum(Gamma_neighbor)
            # for j in range(len(SNR_neighbors)):
            #     gamma += (1 * add_trans[j]) / np.log2(1 + SNR_neighbors[j])
        return constant * 2 + (float(trans_size) / full_size) * gamma

def communication_cost_quan_multiple(node, iter, full_size, trans_size, channel_quality):
    if trans_size == 0:
        return 0.0
    else:
        constant = 8 / full_size  # beta
        # print('neighbors', channel_quality)
        if channel_quality is None:
            rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))
            SNR_0 = rs.chisquare(df=2)
            gamma = 1 / np.log2(1 + SNR_0)
        else:
            neighbors, add_trans = channel_quality
            neighbors = np.setdiff1d(neighbors, node)
            SNR = []
            Gamma_neighbor = []
            for i in range(len(neighbors)):
                rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456 + neighbors[i])))
                SNR.append(rs.chisquare(df=2))
                gamma_neighbor = []
                for k in range(len(add_trans[i])):
                    rs1 = RandomState(MT19937(SeedSequence(iter * 800 + neighbors[i] + 23456 + add_trans[i][k])))
                    gamma_neighbor.append(1 / np.log2(1 + rs1.chisquare(df=2)))
                Gamma_neighbor.append(sum(gamma_neighbor))
            SNR = min(SNR)
            gamma = 1 / np.log2(1 + SNR)
            gamma += sum(Gamma_neighbor)
            # for j in range(len(SNR_neighbors)):
            #     gamma += (1 * add_trans[j]) / np.log2(1 + SNR_neighbors[j])
        return constant*2 + (float(trans_size) / full_size) * gamma

class Compression(abc.ABC):
    def __init__(self, node):
        self.node = node

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, channel_quality):  # w_tmp is gradient this time
        if w_tmp is None:
            w_tmp = w_residual  # w_residual is e_t
        else:
            w_tmp += w_residual

        trans_indices, not_trans_indices, trans_bits = self._get_trans_indices(iter, w_tmp, channel_quality)

        w_tmp_residual = copy.deepcopy(w_tmp)
        w_tmp[not_trans_indices] = 0  # transfer vector v_t, sparse vector
        w_tmp_residual -= w_tmp  # accumulate the residual for not transmit bits

        return w_tmp, w_tmp_residual

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        raise NotImplementedError()  #TODO: What does this mean?

class Compression_1(abc.ABC):
    def __init__(self, node):
        self.node = node

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, channel_quality):  # w_tmp is gradient this time
        if w_tmp is None:
            w_tmp = w_residual  # w_residual is e_t
        else:
            w_tmp += w_residual

        trans_indices, not_trans_indices = self._get_trans_indices(iter, w_tmp, channel_quality)

        w_tmp_residual = copy.deepcopy(w_tmp)
        w_tmp[not_trans_indices] = 0  # transfer vector v_t, sparse vector
        w_tmp_residual -= w_tmp  # accumulate the residual for not transmit bits

        return w_tmp, w_tmp_residual

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        raise NotImplementedError()  #TODO: What does this mean?

class Compression_Q(abc.ABC):
    def __init__(self, node):
        self.node = node

    def get_trans_bits_and_residual(self, iter, w_tmp, device, w_residual, channel_quality):  # w_tmp is gradient this time
        if w_tmp is None:
            w_tmp = w_residual  # w_residual is e_t
        else:
            w_tmp += w_residual

        quantize_value, residual_value = self._get_trans_indices(iter, w_tmp, channel_quality)

        return quantize_value, residual_value

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        raise NotImplementedError()  #TODO: What does this mean?

# "Chose Different Compression Method"
class Lyapunov_compression(Compression):
    def __init__(self, node, avg_comm_cost, V, W):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.V = V
        self.queue = W  # Initial queue length

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        # full_size = w_tmp.size()[0]
        full_size = w_tmp.shape[0]
        bt_square = torch.square(w_tmp)
        Bt = torch.sum(bt_square)
        bt_sq_sort, bt_sq_sort_indices = torch.sort(bt_square, descending=True)

        no_transmit_penalty = self.V * torch.sum(bt_square) - self.queue * self.avg_comm_cost
        cost_delta = self.queue * (communication_cost(self.node, iter, full_size, 2) - communication_cost(self.node, iter, full_size, 1))  # equal to gamma_t * PHI_t(queue at time t)

        tmp = torch.arange(bt_square.shape[0], device=device)
        tmp2 = tmp[self.V * bt_sq_sort <= cost_delta]
        if len(tmp2) > 0:
            j = tmp2[0]
            # print(self.node, len(tmp2), j)
        else:
            j = full_size

        drift_plus_penalty = self.V * torch.sum(bt_sq_sort[j:]) + self.queue * (communication_cost(self.node, iter, full_size, j) - self.avg_comm_cost)

        if drift_plus_penalty < no_transmit_penalty:
            trans_bits = j
        else:
            trans_bits = 0
        self.queue += communication_cost(self.node, iter, full_size, trans_bits) - self.avg_comm_cost
        self.queue = max(0.001, self.queue)  # Not allow to have the negative queues, set to very small one
        return bt_sq_sort_indices[:trans_bits], bt_sq_sort_indices[trans_bits:], trans_bits

class Lyapunov_compression_1(Compression_1):
    def __init__(self, node, avg_comm_cost, V, W):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.V = V
        self.queue = W  # Initial queue length

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        # full_size = w_tmp.size()[0]
        full_size = w_tmp.shape[0]
        bt_square = torch.square(w_tmp)
        Bt = torch.sum(bt_square)
        bt_sq_sort, bt_sq_sort_indices = torch.sort(bt_square, descending=True)
        # print(self.node, channel_quality)

        no_transmit_penalty = self.V * torch.sum(bt_square) - self.queue * self.avg_comm_cost
        cost_delta = self.queue * (communication_cost(self.node, iter, full_size, 2) - communication_cost(self.node, iter, full_size, 1))  # equal to gamma_t * PHI_t(queue at time t)

        tmp = torch.arange(bt_square.shape[0], device=device)
        tmp2 = tmp[self.V * bt_sq_sort <= cost_delta]
        if len(tmp2) > 0:
            j = tmp2[0]
        else:
            j = full_size

        drift_plus_penalty = self.V * torch.sum(bt_sq_sort[j:]) + self.queue * (
                    communication_cost_multiple(self.node, iter, full_size, j, channel_quality) - self.avg_comm_cost)
        # drift_plus_penalty = self.V * torch.sum(bt_sq_sort[j:]) + self.queue * (
        #         communication_cost(self.node, iter, full_size, j) - self.avg_comm_cost)

        if drift_plus_penalty < no_transmit_penalty:
            trans_bits = j
        else:
            trans_bits = 0
        self.queue += communication_cost(self.node, iter, full_size, trans_bits) - self.avg_comm_cost
        self.queue = max(0.001, self.queue)  # Not allow to have the negative queues, set to very small one
        return bt_sq_sort_indices[:trans_bits], bt_sq_sort_indices[trans_bits:]

class Lyapunov_compression_Q(Compression_Q):
    def __init__(self, node, avg_comm_cost, V, W, max_value, min_value):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.V = V
        self.queue = W  # Initial queue length
        self.max_value = max_value
        self.min_value = min_value

    def _get_trans_indices(self, iter, w_tmp, channel_quality):

        full_size = 32
        no_trans_cost = self.V * torch.sum(torch.square(w_tmp)) - self.queue * self.avg_comm_cost
        residual_value = w_tmp
        for trans_bits in [4, 6, 8, 10]:  # [4, 6, 8, 10, 12]
            scale = 2 ** trans_bits - 1
            step = (self.max_value - self.min_value) / scale

            centroids = []
            value = self.min_value
            centroids.append(value)
            while len(centroids) < 2 ** trans_bits:
                value = value + step
                centroids.append(value)

            centroids = torch.tensor(centroids).to(device)
            distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(centroids, (-1, 1)))
            assignments = torch.argmin(distances, dim=1)

            quantize_value = torch.index_select(input=torch.tensor(centroids), dim=0, index=assignments)
            # trans_cost = self.V * torch.sum(torch.square((w_tmp - quantize_value))) + self.queue * (communication_cost_quan(self.node, iter, full_size, trans_bits) - self.avg_comm_cost)

            trans_cost = self.V * torch.sum(torch.square((w_tmp - quantize_value))) + self.queue * (communication_cost_quan_multiple(self.node, iter, full_size, trans_bits, channel_quality) - self.avg_comm_cost)

            if trans_cost < no_trans_cost:
                trans_bits = trans_bits
                quantize_value = quantize_value
                break
            else:
                trans_bits = 0
                quantize_value = torch.zeros_like(w_tmp)
        residual_value -= quantize_value

        self.queue += communication_cost_quan(self.node, iter, full_size, trans_bits) - self.avg_comm_cost
        self.queue = max(0.001, self.queue)  # Not allow to have the negative queues, set to very small one
        return quantize_value, residual_value


class Fixed_Compression(Compression):
    def __init__(self, node, avg_comm_cost, ratio=1.0):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.ratio = ratio

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        full_size = w_tmp.size()[0]
        bt_square = torch.square(w_tmp)
        bt_square_sorted, bt_sorted_indices = torch.sort(bt_square, descending=True)

        no_trans_cost = communication_cost(self.node, iter, full_size, 0)
        if no_trans_cost > 0:
            raise Exception('No transmit cost should be zero')

        k = int(full_size * self.ratio)
        if k > torch.count_nonzero(bt_square).item():
            k = torch.count_nonzero(bt_square).item()
        trans_cost = communication_cost(self.node, iter, full_size, k)

        if trans_cost > 0:
            p_trans = min(1.0, self.avg_comm_cost / trans_cost)
        else:
            p_trans = 1.0

        if np.random.binomial(1, p_trans) == 1:
            trans_bits = k
        else:
            trans_bits = 0
        return bt_sorted_indices[:trans_bits], bt_sorted_indices[trans_bits:], trans_bits

class Top_k(Compression):
    def __init__(self, node, avg_comm_cost, ratio=1.0):
        super().__init__(node)
        self.ratio = ratio

    def _get_trans_indices(self, iter, w_tmp, channel_quality):
        full_size = w_tmp.size()[0]
        # print(iter, full_size, self.ratio)
        bt_square = torch.square(w_tmp)
        bt_square_sorted, bt_sorted_indices = torch.sort(bt_square, descending=True)
        trans_bits = int(self.ratio * full_size)
        return bt_sorted_indices[:trans_bits], bt_sorted_indices[trans_bits:], trans_bits

class Quantization(abc.ABC):
    def __init__(self, num_bits=8, max_value=0, min_value=0):
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1
        self.max_value = max_value
        self.min_value = min_value
        if self.max_value == self.min_value == 0:
            raise Exception('Please set the max and min value for quantization')
        self._initialization()

    def _initialization(self):
        step = (self.max_value - self.min_value) / self.scale

        quantization = []
        value = self.min_value
        quantization.append(value)
        while len(quantization) < 2 ** self.num_bits:
            value = value + step
            quantization.append(value)
        self.quantization = torch.tensor(quantization)

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, channel_quality):
        if w_tmp is None:
            w_tmp = w_residual  # w_residual is e_t
        else:
            w_tmp += w_residual

        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(self.quantization, (-1, 1)))
        assignments = torch.argmin(distances, dim=1)

        w_tmp_quantized = torch.index_select(input=torch.tensor(self.quantization), dim=0, index=assignments)
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

# class Quantization(abc.ABC):
#     def __init__(self, num_bits=4, max_value=0, min_value=0):
#         self.num_bits = num_bits
#         self.scale = 2**self.num_bits - 1
#         self.max_value = max_value
#         self.min_value = min_value
#         self.max = []
#         self.min = []
#
#     def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device, channel_quality):
#         if w_tmp is None:
#             w_tmp = w_residual  # w_residual is e_t
#         else:
#             w_tmp += w_residual
#         max_value = torch.max(w_tmp)
#         min_value = torch.min(w_tmp)
#         self.max.append(max_value)
#         self.min.append(min_value)
#
#         step = (max_value - min_value) / self.scale
#
#         centroids = []
#         value = min_value
#         centroids.append(value)
#         while len(centroids) < 2 ** self.num_bits:
#             value = value + step
#             centroids.append(value)
#
#         centroids = torch.tensor(centroids).to(device)
#         distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(centroids, (-1, 1)))
#         assignments = torch.argmin(distances, dim=1)
#
#         w_tmp_quantized = torch.tensor([centroids[i] for i in assignments])
#         w_residual = w_tmp - w_tmp_quantized
#         return w_tmp_quantized, w_residual
