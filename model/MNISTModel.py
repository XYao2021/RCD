import torch
from torch import nn

class MNISTModel(nn.Module):  # Improve the model V0 with nonlinear activation function nn.Relu()
    def __init__(self, input_shape=784,
                 output_shape=10,
                 hidden_units=50):
        super().__init__()
        self.layer_1 = nn.Sequential(nn.Flatten(),  # Equal to x.view(-1, 784)
                                     nn.Linear(in_features=input_shape, out_features=hidden_units),
                                     nn.ReLU())
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=output_shape)

        @torch.no_grad()
        def initial_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                # nn.init.zeros_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.zeros_(m.bias)

        def zero_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.zeros_(m.bias)

        def zero_bias(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.zeros_(m.bias)

        self.layer_1.apply(initial_weights)
        self.layer_2.apply(initial_weights)

    def forward(self, x):
        x = self.layer_1(x)
        return self.layer_2(x)

