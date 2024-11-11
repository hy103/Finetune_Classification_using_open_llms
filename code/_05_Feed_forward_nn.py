import torch 
import torch.nn as nn
from _03_GELU_activation_function import GELU

class FeedForward_network(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"],  4*cfg["emb_dim"]),

            GELU(),

            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

def main():
    pass   
if __name__ == '__main__':
    main()