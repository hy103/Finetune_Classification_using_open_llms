import torch.nn as nn
import torch
from _03_GELU_activation_function import GELU


class Shortconnections(nn.Module):
    def __init__(self, layer_size, use_shortcut):
        super().__init__()

        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_size[0], layer_size[1]), GELU()),
            nn.Sequential(nn.Linear(layer_size[1], layer_size[2]), GELU()),
            nn.Sequential(nn.Linear(layer_size[2], layer_size[3]), GELU()),
            nn.Sequential(nn.Linear(layer_size[3], layer_size[4]), GELU()),
            nn.Sequential(nn.Linear(layer_size[4], layer_size[5]), GELU())]
        )

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x+layer_output

            else :
                x = layer_output
        return x
    

def main():
    pass   
if __name__ == '__main__':
    main()