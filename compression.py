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


def communication_cost(node, iter, full_size, trans_size):
    if trans_size == 0:
        return 0.0
    else:
        rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))

        if node >= 0:
            constant = 0.05  # beta
            return constant + float(trans_size)/full_size / np.log2(1 + rs.chisquare(df=2))
        else:
            constant = 0.01  # smaller at server
            return constant + 0.2 * float(trans_size)/full_size / np.log2(1 + rs.chisquare(df=2))

def communication_cost_quan(node, iter, full_size, trans_size):
    if trans_size == 0:
        return 0.0
    else:
        rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))

        if node >= 0:
            constant = 16  # beta
            return constant + trans_size * np.log2(1 + rs.chisquare(df=2))
        else:
            constant = 16  # smaller at server
            return constant + 0.2 * trans_size * np.log2(1 + rs.chisquare(df=2))

def communication_cost_1(node, iter, full_size, trans_size):
    if trans_size == 0:
        return 0.0
    else:
        rs = RandomState(MT19937(SeedSequence(iter * 801 + node + 23455)))

        if node >= 0:
            constant = 0.05  # beta
            return constant + float(trans_size)/full_size / np.log2(1 + rs.chisquare(df=2))
        else:
            constant = 0.01  # smaller at server
            return constant + 0.2 * float(trans_size)/full_size / np.log2(1 + rs.chisquare(df=2))

def communication_cost_quan_1(node, iter, full_size, trans_size):
    if trans_size == 0:
        return 0.0
    else:
        rs = RandomState(MT19937(SeedSequence(iter * 801 + node + 23455)))

        if node >= 0:
            constant = 16  # beta
            return constant + trans_size * np.log2(1 + rs.chisquare(df=2))
        else:
            constant = 16  # smaller at server
            return constant + 0.2 * trans_size * np.log2(1 + rs.chisquare(df=2))

class Compression(abc.ABC):
    def __init__(self, node):
        self.node = node

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual):  # w_tmp is gradient this time
        if w_tmp is None:
            w_tmp = w_residual  # w_residual is e_t
        else:
            w_tmp += w_residual

        trans_indices, not_trans_indices, trans_bits = self._get_trans_indices(iter, w_tmp)

        w_tmp_residual = copy.deepcopy(w_tmp)
        w_tmp[not_trans_indices] = 0  # transfer vector v_t, sparse vector
        w_tmp_residual -= w_tmp  # accumulate the residual for not transmit bits

        return w_tmp, w_tmp_residual

    def _get_trans_indices(self, iter, w_tmp):
        raise NotImplementedError()  #TODO: What does this mean?

class Compression_Q(abc.ABC):
    def __init__(self, node):
        self.node = node

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual):  # w_tmp is gradient this time
        if w_tmp is None:
            w_tmp = w_residual  # w_residual is e_t
        else:
            w_tmp += w_residual

        quantize_value, residual_value = self._get_trans_indices(iter, w_tmp)

        return quantize_value, residual_value

    def _get_trans_indices(self, iter, w_tmp):
        raise NotImplementedError()  #TODO: What does this mean?

# "Chose Different Compression Method"
class Lyapunov_compression(Compression):
    def __init__(self, node, avg_comm_cost, V, W):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.V = V
        self.queue = W  # Initial queue length

    def _get_trans_indices(self, iter, w_tmp):
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
        drift_plus_penalty_1 = self.V * torch.sum(bt_sq_sort[j:]) + self.queue * (communication_cost_1(self.node, iter, full_size, j) - self.avg_comm_cost)

        if drift_plus_penalty < no_transmit_penalty and drift_plus_penalty_1 < no_transmit_penalty:
            trans_bits = j
        else:
            trans_bits = 0
        self.queue += communication_cost(self.node, iter, full_size, trans_bits) - self.avg_comm_cost
        self.queue = max(0.001, self.queue)  # Not allow to have the negative queues, set to very small one
        return bt_sq_sort_indices[:trans_bits], bt_sq_sort_indices[trans_bits:], trans_bits

class Fixed_Compression(Compression):
    def __init__(self, node, avg_comm_cost, ratio=1.0):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.ratio = ratio

    def _get_trans_indices(self, iter, w_tmp):
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

    def _get_trans_indices(self, iter, w_tmp):
        full_size = w_tmp.size()[0]
        bt_square = torch.square(w_tmp)
        bt_square_sorted, bt_sorted_indices = torch.sort(bt_square, descending=True)
        trans_bits = int(self.ratio * full_size)
        return bt_sorted_indices[:trans_bits], bt_sorted_indices[trans_bits:], trans_bits

# class Rand_k(Compression):
#     def __init__(self, node, avg_comm_cost, ratio=1.0):
#         super().__init__(node)
#         self.ratio = ratio
#
#     def _get_trans_indices(self, iter, w_tmp):
#         np.random.seed()
#         full_size = w_tmp.size()[0]
#         all_indices = np.arange(full_size, dtype=int)
#         send_indices = np.random.choice(all_indices, int(full_size*self.ratio), replace=False)
#         # print(self.node, send_indices)
#         not_send_indices = np.setdiff1d(all_indices, send_indices)
#         return send_indices, not_send_indices

class Quantization(abc.ABC):
    def __init__(self, num_bits=4):
        self.num_bits = num_bits
        self.scale = 2**self.num_bits - 1

    def get_trans_bits_and_residual(self, iter, w_tmp, w_residual, device):
        if w_tmp is None:
            w_tmp = w_residual  # w_residual is e_t
        else:
            w_tmp += w_residual
        max_value = torch.max(w_tmp)
        min_value = torch.min(w_tmp)
        step = (max_value - min_value) / self.scale

        centroids = []
        value = min_value
        centroids.append(value)
        while len(centroids) < 2 ** self.num_bits:
            value = value + step
            centroids.append(value)

        centroids = torch.tensor(centroids).to(device)
        distances = torch.cdist(torch.reshape(w_tmp, (-1, 1)), torch.reshape(centroids, (-1, 1)))
        assignments = torch.argmin(distances, dim=1)
        # w_tmp = torch.reshape(w_tmp, (-1, 1))
        #
        # kmeans = KMeans(n_clusters=2 ** self.num_bits)
        # kmeans.fit(w_tmp)
        # centers = kmeans.cluster_centers_
        # labels = kmeans.labels_

        w_tmp_quantized = torch.tensor([centroids[i] for i in assignments])
        w_residual = w_tmp - w_tmp_quantized
        return w_tmp_quantized, w_residual

class Lyapunov_compression_Q(Compression_Q):
    def __init__(self, node, avg_comm_cost, V, W):
        super().__init__(node)
        self.avg_comm_cost = avg_comm_cost
        self.V = V
        self.queue = W  # Initial queue length

    def _get_trans_indices(self, iter, w_tmp):
        # full_size = w_tmp.size()[0]
        full_size = w_tmp.shape[0]
        bt_square = torch.square(w_tmp)
        residual_value = w_tmp

        no_trans_cost = self.V * torch.sum(torch.square(w_tmp)) - self.queue * self.avg_comm_cost
        for trans_bits in range(1, 33):
            scale = 2 ** trans_bits - 1
            quantize_value = torch.round(w_tmp * scale) / scale
            trans_cost = self.V * torch.sum(torch.square((w_tmp - quantize_value))) + self.queue * (communication_cost_quan(self.node, iter, full_size, trans_bits) - self.avg_comm_cost)
            trans_cost_1 = self.V * torch.sum(torch.square((w_tmp - quantize_value))) + self.queue * (communication_cost_quan_1(self.node, iter, full_size, trans_bits) - self.avg_comm_cost)
            # print(iter, self.node, trans_bits, trans_cost, no_trans_cost, self.V * torch.sum(torch.square(w_tmp)), self.V * torch.sum(torch.square((w_tmp - quantize_value))))

            if trans_cost < no_trans_cost and trans_cost_1 < no_trans_cost:
                trans_bits = trans_bits
                quantize_value = quantize_value
                break
            else:
                trans_bits = 0
                quantize_value = torch.zeros_like(w_tmp)
        residual_value -= quantize_value
        # print(iter, self.node, trans_bits, trans_cost, no_trans_cost)

        self.queue += communication_cost_quan(self.node, iter, full_size, trans_bits) - self.avg_comm_cost
        self.queue = max(0.001, self.queue)  # Not allow to have the negative queues, set to very small one
        return quantize_value, residual_value
