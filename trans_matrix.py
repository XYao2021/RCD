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
        self.matrix = []
        self.factor = 0

        if seed is not None:
            np.random.seed(seed)
        if network == 'Ring':
            self._Ring_network()
        elif network == 'Torus':
            self._Torus_network()
        elif network == 'Cyclic':
            self._Cyclic_network()
        else:
            self.generate_symmetric_matrix()

    def _Cyclic_network(self):  # TODO: Change the neighbors algorithm
        clients = np.arange(self.nodes, dtype=int)

        for i in range(self.nodes):
            neighbor = np.arange(i, i + self.num_neighbor)
            if i + self.num_neighbor + 1 > self.nodes:
                neighbor %= self.nodes

            self.neighbors.append(neighbor)
        self.factor = 1 / len(self.neighbors[0])

    def _Torus_network(self):  # 2-D Torus network topology
        N = int(np.sqrt(self.nodes))
        torus = np.arange(self.nodes, dtype=int).reshape((N, N))

        for i in range(self.nodes):
            x, y = np.where(torus == i)
            self.neighbors.append(
                [torus[x, y][0], torus[(x + 1) % N, y][0], torus[x, (y + 1) % N][0], torus[(x - 1) % N, y][0],
                 torus[x, (y - 1) % N][0]])
        self.factor = 1 / len(self.neighbors[0])

    def _Ring_network(self):
        for i in range(self.nodes):
            self.neighbors.append([(i-1) % self.nodes, i, (i+1) % self.nodes])
        self.factor = 1 / len(self.neighbors[0])

    def generate_symmetric_matrix(self):
        upper = int(self.nodes / 2) - 2
        bottom = 1
        if self.num_neighbor + 1 > int(self.nodes / 2) - 1:
            print('Number should be in range [{}, {}]'.format(bottom, upper))
            raise Exception('Invalid neighbor number')
        matrix = np.ones((self.nodes,), dtype=int)
        while True:
            org_matrix = np.diag(matrix)
            org_target = np.arange(self.nodes, dtype=int)
            for i in range(self.nodes):
                if np.count_nonzero(org_matrix[i]) < self.num_neighbor + 1:
                    if np.count_nonzero(org_matrix[i]) < self.num_neighbor + 1 and np.count_nonzero(org_matrix.transpose()[i]) < self.num_neighbor + 1:
                        target = np.setdiff1d(org_target, i)
                        target_set = []
                        for k in range(len(target)):
                            if np.count_nonzero(org_matrix[target[k]]) < self.num_neighbor + 1:
                                target_set.append(target[k])
                        if self.num_neighbor + 1 - int(np.count_nonzero(org_matrix[i])) <= len(target_set):
                            target = np.random.choice(target_set, self.num_neighbor + 1 - int(np.count_nonzero(org_matrix[i])), replace=False)
                        for j in range(len(target)):
                            org_matrix[i][target[j]] = 1
                            org_matrix.transpose()[i][target[j]] = 1
                else:
                    pass
            if np.count_nonzero(np.array([np.count_nonzero(org_matrix[i]) for i in range(self.nodes)]) - (self.num_neighbor + 1)) == 0:
                break
        self.matrix = org_matrix
        for i in range(len(org_matrix)):
            neighbors = [index for index in range(len(org_matrix[i])) if org_matrix[i][index] == 1]
            self.neighbors.append(neighbors)
        self.factor = 1/len(self.neighbors[0])

    def Average(self, Update_Vector):
        Update_Avg = []
        for i in range(self.nodes):
            updates = torch.zeros_like(Update_Vector[0])
            for j in range(len(self.neighbors[i])):
                updates += Update_Vector[self.neighbors[i][j]]
            updates = torch.div(updates, torch.tensor(len(self.neighbors[0])))
            Update_Avg.append(updates)
        return Update_Avg

    def Average_CHOCO(self, Update_Vector):
        Update_Avg = []
        for i in range(self.nodes):
            updates = torch.zeros_like(Update_Vector[0])
            # neighbor = np.setdiff1d(self.neighbors[i], i)
            for j in self.neighbors[i]:
                updates += self.factor * (Update_Vector[j] - Update_Vector[i])
            Update_Avg.append(updates)
        return Update_Avg

    def Average_ECD(self, Update_Vector):
        Update_Avg = []
        for i in range(self.nodes):
            updates = torch.zeros_like(Update_Vector[0])
            neighbors = self.neighbors[i]
            # neighbors = np.setdiff1d(self.neighbors[i], i)
            for j in range(len(neighbors)):
                updates += self.factor*Update_Vector[neighbors[j]]
            # updates = torch.div(updates, torch.tensor(len(self.neighbors[0])))
            Update_Avg.append(updates)
        return Update_Avg

    # def Average_DCD(self, Update_Vector):
    #     Update_Avg = []
    #     for i in range(self.nodes):
    #         updates = torch.zeros_like(Update_Vector[0])
    #         for j in range(len(self.neighbors[i])):
    #             updates += (Update_Vector[self.neighbors[i][j]] - Update_Vector[i])
    #         updates = torch.div(updates, torch.tensor(len(self.neighbors[0])))
    #         Update_Avg.append(updates)
    #     return Update_Avg
