import torch
import torch.nn as nn
class Multihead_attention(nn.Module):
    def __init__(self, d_in, d_out, context_len, num_heads, dropout, qkv_bias):
        super().__init__()

        assert (d_out % num_heads == 0), \
"d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out
        self.head_dim = d_out//num_heads
        self.context_len = context_len
        self.dropout = dropout
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask",
                             torch.triu(torch.ones(self.context_len , self.context_len), diagonal= 1))
        

    def forward(self, x):
        Query = self.W_query(x)
        keys = self.W_keys(x)
        values = self.W_values(x)

        num_batches, num_tokens, d_in = x.shape

        keys = keys.view(num_batches, num_tokens, self.num_heads, self.head_dim)
        Query = Query.view(num_batches, num_tokens, self.num_heads, self.head_dim)
        values = values.view(num_batches, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        Query = Query.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = Query @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Correct multiplication and reshaping
        context_vec = (attention_weights @ values).transpose(1, 2)

        # Now, reshape to combine heads and head_dim into d_out
        context_vec = context_vec.contiguous().view(num_batches, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)

        return context_vec
    

def main():
    pass   
if __name__ == '__main__':
    main()