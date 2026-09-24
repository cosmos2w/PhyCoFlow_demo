import torch
import torch.nn as nn

# Fully Connected Neural Network (FCNN) class to dynamically build networks
class FCN(nn.Module):
    """
    Fully Connected Neural Network (FCNN) class to dynamically build networks
    Args:
    - architecture (list): List of integers where each integer represents the number of neurons in a layer
    - activation_fn (torch.nn.Module): Activation function to apply after each layer

    Example:
    architecture = [2, 64, 64, 1]
    activation_fn = nn.ReLU
    fcn = FCN(architecture, activation_fn)
    """
    def __init__(self, architecture, activation_fn=nn.ReLU):
        super(FCN, self).__init__()
        layers = []
        for i in range(len(architecture) - 1):
            # For the last layer, set bias=False
            if i == len(architecture) - 2:
                layers.append(nn.Linear(architecture[i], architecture[i + 1], bias=True))
            else:
                layers.append(nn.Linear(architecture[i], architecture[i + 1]))
                layers.append(activation_fn())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
