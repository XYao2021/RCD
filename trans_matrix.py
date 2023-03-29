import numpy as np
import copy
import torch

class Transform:
    def __init__(self, num_nodes, num_neighbors, seed=None, network=None):
        super().__init__()
        self.nodes = num_nodes
        self.num_neighbor = num_neighbors
        self.neighbors = []
        self.network = network

        if seed is not None:
            np.random.seed(seed)
        if network == 'Ring':
            self._Ring_network()
        elif network == 'Torus':
            self._Torus_network()
        else:
            self._get_neighbors()

        self.factor = 1/len(self.neighbors[0])

    def _get_neighbors(self):  # TODO: Change the neighbors algorithm

        # if self.nodes % self.num_neighbor != 0:
        #     raise Exception('num_neighbor should be factor of num_nodes')
        clients = np.arange(self.nodes, dtype=int)

        for i in range(self.nodes):
            # neighbor = np.setdiff1d(clients, i)
            # neighbor = np.random.choice(neighbor, self.num_neighbor, replace=False)
            # neighbor = np.append(neighbor, i)
            neighbor = np.arange(i, i + self.num_neighbor)
            if i + self.num_neighbor + 1 > self.nodes:
                neighbor %= self.nodes
            # neighbor = np.arange((i - self.num_neighbor) % self.nodes, i)
            # print(neighbor)

            self.neighbors.append(neighbor)

    def _Torus_network(self):  # 2-D Torus network topology
        N = int(np.sqrt(self.nodes))
        torus = np.arange(self.nodes, dtype=int).reshape((N, N))

        for i in range(self.nodes):
            x, y = np.where(torus == i)
            self.neighbors.append(
                [torus[x, y][0], torus[(x + 1) % N, y][0], torus[x, (y + 1) % N][0], torus[(x - 1) % N, y][0],
                 torus[x, (y - 1) % N][0]])

    def _Ring_network(self):
        for i in range(self.nodes):
            self.neighbors.append([(i-1) % self.nodes, i, (i+1) % self.nodes])

    def Average(self, Update_Vector):
        Update_Avg = []
        # update_vector = copy.deepcopy(Update_Vector)
        for i in range(self.nodes):
            updates = torch.zeros_like(Update_Vector[0])
            # print(self.neighbors[i])
            for j in range(len(self.neighbors[i])):
                updates += Update_Vector[self.neighbors[i][j]]
            updates = torch.div(updates, torch.tensor(len(self.neighbors[0])))
            Update_Avg.append(updates)
        return Update_Avg

    # def Average(self, Update_Vector):
    #     Client_update = [[] for _ in range(len(Update_Vector))]
    #     for i in range(self.nodes):
    #         for j in range(len(self.neighbors[i])):
    #             Client_update[i].append(Update_Vector[self.neighbors[i][j]])
    #     Final = []
    #     for n in range(self.nodes):
    #         for m in range(len(Client_update[n])):
    #             if m == 0:
    #                 updates = Client_update[n][m]
    #             else:
    #                 updates += Client_update[n][m]
    #         Final.append(self.factor * updates)
    #     return Final


# a = Transform(num_nodes=100, num_neighbors=10, seed=2, network='Torus')
# print(a.neighbors)
