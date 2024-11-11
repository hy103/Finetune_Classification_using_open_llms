import torch
import torch.nn as nn
import math

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):

       return 0.5*x*(1+ torch.tanh(torch.sqrt(2/(torch.Tensor([math.pi])))*(x + 0.044715*(torch.pow(x, 3)))))
    
def main():
    pass   
if __name__ == '__main__':
    main()