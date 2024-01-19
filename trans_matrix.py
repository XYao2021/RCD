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
        self.add_trans = []

        if seed is not None:
            np.random.seed(seed)
        if network == 'Ring':
            # self._Ring_network()
            self.random_ring_network()
        elif network == 'Torus':
            self._Torus_network()
        elif network == 'Cyclic':
            self._Cyclic_network()
        else:
            self.generate_symmetric_matrix()
        self.Add_trans()

    def _Cyclic_network(self):  # TODO: Change the neighbors algorithm
        # clients = np.arange(self.nodes, dtype=int)
        clients = [i for i in range(self.nodes)]

        if self.num_neighbor % 2 != 0 or self.num_neighbor > self.nodes / 2:
            raise Exception('Number of connected nodes should be even number')

        for i in range(self.nodes):
            if i - int(self.num_neighbor / 2) < 0:
                neighbors = clients[i - int(self.num_neighbor / 2):] + clients[: i + int(self.num_neighbor / 2) + 1]
            elif i + int(self.num_neighbor / 2) > self.nodes - 1:
                neighbors = clients[i - int(self.num_neighbor / 2):] + clients[:(i + int(self.num_neighbor / 2) - self.nodes) + 1]
            else:
                neighbors = clients[i - int(self.num_neighbor / 2): i + int(self.num_neighbor / 2) + 1]

            self.neighbors.append(neighbors)
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
        nodes = self.nodes

        matrix = np.ones((nodes,), dtype=int)
        conn_matrix = np.diag(matrix)
        for i in range(self.nodes):
            connected = [(i-1) % self.nodes, i, (i+1) % self.nodes]
            for j in connected:
                conn_matrix[i][j] = 1
                conn_matrix.transpose()[i][j] = 1
            self.neighbors.append(connected)
        self.factor = 1 / len(self.neighbors[0])
        self.matrix = conn_matrix

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

    def random_ring_network(self):
        nodes = self.nodes
        num_neighbor = 2
        self.num_neighbor = num_neighbor

        matrix = np.ones((nodes,), dtype=int)
        conn_matrix = np.diag(matrix)

        clients = np.arange(nodes, dtype=int)
        selected = [0]

        targets = np.setdiff1d(clients, clients[0])
        target = np.random.choice(targets, num_neighbor, replace=False)
        selected += list(target)
        for j in range(len(target)):
            conn_matrix[clients[0]][target[j]] = 1
            conn_matrix.transpose()[clients[0]][target[j]] = 1

        sub_target = list(target)
        time = 0
        while len(sub_target) != 0:
            new_target = []
            for node in sub_target:
                sub_target_1 = np.setdiff1d(clients, selected)
                sub_target_1 = np.random.choice(sub_target_1, 1, replace=False)
                new_target.append(list(sub_target_1)[0])
                selected += list(sub_target_1)
                conn_matrix[node][list(sub_target_1)[0]] = 1
                conn_matrix.transpose()[node][list(sub_target_1)[0]] = 1
            if len(selected) == nodes - 1:
                final_target = np.setdiff1d(clients, selected)
                sub_target = [i for i in range(nodes) if
                              np.count_nonzero(conn_matrix[i]) < num_neighbor + 1 and np.count_nonzero(
                                  conn_matrix.transpose()[i]) < num_neighbor + 1]
                sub_target = np.setdiff1d(sub_target, final_target)
                for client in sub_target:
                    conn_matrix[client][list(final_target)[0]] = 1
                    conn_matrix[list(final_target)[0]][client] = 1
                sub_target = []
            else:
                sub_target = new_target
            if len(sub_target) == 0:
                break

        neighbors = [[] for i in range(nodes)]
        for i in range(nodes):
            for j in range(len(conn_matrix[i])):
                if conn_matrix[i][j] == 1:
                    neighbors[i].append(j)
        self.matrix = conn_matrix
        self.neighbors = neighbors
        self.factor = 1 / len(self.neighbors[0])

    def Add_trans(self):
        conn_neighbor = []
        for i in range(self.nodes):
            connected = np.setdiff1d(self.neighbors[i], i)
            conn_neighbor.append(connected)

        Add_trans = []
        for i in range(self.nodes):
            neighbor = conn_neighbor[i]
            add_trans = []
            for j in range(self.num_neighbor):
                remain = np.setdiff1d(conn_neighbor[conn_neighbor[i][j]], i)
                need_trans = []
                for k in range(len(remain)):
                    if i not in conn_neighbor[remain[k]]:
                        need_trans.append(remain[k])
                add_trans.append(need_trans)
            Add_trans.append(add_trans)
        self.add_trans = Add_trans

    def Average(self, Update_Vector):
        Update_Avg = []
        for i in range(self.nodes):
            updates = torch.zeros_like(Update_Vector[0])
            for j in range(len(self.neighbors[i])):
                updates += self.factor * Update_Vector[self.neighbors[i][j]]
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
