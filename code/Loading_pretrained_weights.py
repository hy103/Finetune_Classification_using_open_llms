# import urllib.request

# url = ("https://raw.githubusercontent.com/rasbt/"
# "LLMs-from-scratch/main/ch05/"
# "01_main-chapter-code/gpt_download.py")

# filename = url.split('/')[-1]
# urllib.request.urlretrieve(url, filename)

import torch
from GPT_Model import GPTModel
import numpy as np
from text_token_text import text_to_token_ids, token_ids_to_text
import tiktoken

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
    "Right: {right.shape}"
    )
    return torch.nn.Parameter(torch.tensor(right,  dtype=left.dtype))


def generate(model, idx, max_new_tokens, context_size,
    temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens): #1
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None: #2
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
        )
        if temperature > 0.0: #3
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else: #4
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id: #5
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def load_weights_into_gpt(gpt, params):

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])


    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis =-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight,  q_w.T)
        gpt.trf_blocks[b].att.W_keys.weight =   assign(gpt.trf_blocks[b].att.W_keys.weight,  k_w.T)
        gpt.trf_blocks[b].att.W_values.weight = assign(gpt.trf_blocks[b].att.W_values.weight,  v_w.T)


        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis =-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias,  q_b.T)
        gpt.trf_blocks[b].att.W_keys.bias =   assign(gpt.trf_blocks[b].att.W_keys.bias,  k_b.T)
        gpt.trf_blocks[b].att.W_values.bias = assign(gpt.trf_blocks[b].att.W_values.bias,  v_b.T)


        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias =   assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias   = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[1].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias   = assign(gpt.trf_blocks[1].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].layernorm1.scale = assign(gpt.trf_blocks[b].layernorm1.scale, params["blocks"][b]["ln_1"]["g"])  ## g - gamma
        gpt.trf_blocks[b].layernorm1.shift = assign(gpt.trf_blocks[b].layernorm1.shift, params["blocks"][b]["ln_1"]["b"])  ## b - beta
        gpt.trf_blocks[b].layernorm2.scale = assign(gpt.trf_blocks[b].layernorm2.scale, params["blocks"][b]["ln_2"]["g"])  ## g - gamma
        gpt.trf_blocks[b].layernorm2.shift = assign(gpt.trf_blocks[b].layernorm2.shift, params["blocks"][b]["ln_2"]["b"])  ## b - beta

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.scale, params["b"])
    gpt.out_head.weight  = assign(gpt.out_head.weight, params["wte"])


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = torch.load("model_settings_params.pth")
    model, params = model["model_settings_dict"], model["model_params_dict"]
    #print(model["model_settings_dict"]) ##{'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
    #print(model["model_params_dict"]["wte"].shape)

    model_confifgs = {"gpt2-small (124M)": {"emb_dim" : 768, "n_layers" : 12, "n_heads" : 12},
                    "gpt2-medium (355M)": {"emb_dim" : 1024, "n_layers" : 24, "n_heads" : 16},
                    "gpt2-large (774M)": {"emb_dim" : 1280, "n_layers" : 36, "n_heads" : 20},
                    "gpt2-xl (1558M)": {"emb_dim" : 1600, "n_layers" : 48, "n_heads" : 25}}


    GPT_CONFIG_124M = {
        "vocab_size" : 50257,
        "context_length" : 256,
        "emb_dim" : 768,
        "n_layers" : 12,
        "n_heads" : 12,
        "drop_rate" : 0.1,
        "qkv_bias" : False
    }

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_confifgs[model_name])

    NEW_CONFIG.update({"context_length" : 1024})
    NEW_CONFIG.update({"qkv_bias" : True})
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    load_weights_into_gpt(gpt, params)
    gpt.to(device)


    torch.manual_seed(123)
    token_ids = generate(
        model = gpt,
        idx = text_to_token_ids("Arguments are extremely vulgar", tokenizer).to(device),
        max_new_tokens = 25,
        context_size = NEW_CONFIG["context_length"],
        top_k =50,
        temperature = 1.5
    )

    print("Output text :\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == '__main__':
    main()