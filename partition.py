import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


def get_alpha(node, iter):
    random_variable = RandomState(MT19937(SeedSequence(iter * 1000 + node)))
    return random_variable.random()  # Get the random number between 0 and 1, have plenty other random number generate options

class Lyapunov_Participation:
    def __init__(self, node, average_comp_cost, V, W):
        super().__init__()
        self.node = node
        self.avg_comp_cost = average_comp_cost
        self.V = V
        self.queue = W

    def get_q(self, iter):
        alpha = get_alpha(self.node, iter)
        if self.queue * alpha > 0:
            # print(np.sqrt(self.V), np.sqrt(self.queue * alpha), alpha)
            q_opt = min(1.0, np.sqrt(self.V) / np.sqrt(self.queue * alpha))
        else:
            q_opt = 1.0
        self.queue += q_opt * alpha - self.avg_comp_cost
        self.queue = max(0.0, self.queue)
        return q_opt

class Fixed_Participation:
    def __init__(self, average_comp_cost):
        super().__init__()
        self.avg_comp_cost = average_comp_cost

    def get_q(self, iter):
        mean_cost = 0.5
        return min(1.0, self.avg_comp_cost / mean_cost)


# print(Participation(node=1, average_comp_cost=0.25, V=0.1, W=1, method='Lyapunov').get_Lyapunov_q(10))
