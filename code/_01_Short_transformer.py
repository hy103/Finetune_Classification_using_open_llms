from Multi_head_attention import Multihead_attention
from _03_GELU_activation_function import GELU
from _02_Layer_normalization import LayerNorm
from _05_Feed_forward_nn import FeedForward_network
import torch.nn as nn 
import math
import torch



class Short_transformerblock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = Multihead_attention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_len= cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )

        self.ff = FeedForward_network(cfg)
        self.layernorm1 = LayerNorm(cfg["emb_dim"])
        self.layernorm2 = LayerNorm(cfg["emb_dim"])
        self.drop_short = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):

        shortcut = x
        x = self.layernorm1(x)
        x = self.att(x)
        x = self.drop_short(x)
        x = x+shortcut

        shortcut = x
        x = self.layernorm2(x)
        x = self.att(x)
        x = self.drop_short(x)
        x = x+shortcut

        return x


def main():
    GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
    }

    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)

    st = Short_transformerblock(GPT_CONFIG_124M)

    output = st(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

if __name__ == '__main__':
    main()