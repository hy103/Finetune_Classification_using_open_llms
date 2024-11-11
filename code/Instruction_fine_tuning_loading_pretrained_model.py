from GPT_Model import GPTModel
#from gpt_download import download_and_load_gpt2
from Loading_pretrained_weights import load_weights_into_gpt
import torch
import tiktoken
from Download_create_dataset_dataloader import format_input, custom_collate_fn
from text_token_text import generate, generate_text_simple, text_to_token_ids, token_ids_to_text
import sys

import numpy as np

# Allow numpy objects during unpickling
torch.serialization.add_safe_globals([np.ndarray])

checkpoint = torch.load("model_settings_params.pth")
checkpoint1 = torch.load("datasets_and_tokenizer.pth")
collate_fn_settings = torch.load("collate_fn_settings.pth")

BASE_CONFIG = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "drop_rate" : 0.0,
    "qkv_bias" : True}

model_configs = {
    "gpt2-small (124M)" : {"emb_dim" : 768, "n_layers" : 12, "n_heads" : 12},
    "gpt2-medium (355M)" : {"emb_dim" : 1024, "n_layers" : 24, "n_heads" : 16},
    "gpt2-large (774M)" : {"emb_dim" : 1280, "n_layers" : 36, "n_heads" : 20},
    "gpt2-xl (1550M)" : {"emb_dim" : 1600, "n_layers" : 48, "n_heads" : 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

#checkpoint = torch.load("model_settings_params.pth",  weights_only=True)

# Retrieve settings and params
settings = checkpoint["model_settings_dict"]
params = checkpoint["model_params_dict"]
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()



import torch

# Load datasets and tokenizer
#checkpoint1 = torch.load("datasets_and_tokenizer.pth",  weights_only=True)

train_dataset = checkpoint1["train_dataset"]
val_dataset = checkpoint1["val_dataset"]
test_dataset = checkpoint1["test_dataset"]
train_data = checkpoint1["train_data"]
val_data = checkpoint1["val_data"]
test_data = checkpoint1["test_data"]
tokenizer = checkpoint1["tokenizer"]

# Load collate function settings
#collate_fn_settings = torch.load("collate_fn_settings.pth",  weights_only=True)

device = collate_fn_settings["device"]
allowed_max_length = collate_fn_settings["allowed_max_length"]
pad_token_id = collate_fn_settings["pad_token_id"]
ignore_index = collate_fn_settings["ignore_index"]

# Re-create the customized collate function with the loaded settings
from functools import partial
from torch.utils.data import DataLoader

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=allowed_max_length,
    pad_token_id=pad_token_id,
    ignore_index=ignore_index
)

# Re-create DataLoader instances
train_loader = DataLoader(
    train_dataset,
    batch_size=8,  # Replace with your batch size
    num_workers=0,  # Adjust the number of workers if necessary
    shuffle=True,
    collate_fn=customized_collate_fn,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,  # Replace with your batch size
    num_workers=0,  # Adjust the number of workers if necessary
    shuffle=True,
    collate_fn=customized_collate_fn,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,  # Replace with your batch size
    num_workers=0,  # Adjust the number of workers if necessary
    shuffle=True,
    collate_fn=customized_collate_fn,
    drop_last=True
)

# Now, you can use train_loader, val_loader, and test_loader for training or evaluation
input_text = format_input(val_data[0])
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate(
    model = model,
    idx = text_to_token_ids(input_text, tokenizer),
    max_new_tokens=15,
    context_size= BASE_CONFIG["context_length"],
    eos_id = 50256
)

generated_text = token_ids_to_text(token_ids, tokenizer)

response_text = generated_text[len(input_text):].strip()
print(response_text)


from GPT_training import (
calc_loss_loader,
train_model_simple
)

model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(
    val_loader, model, device, num_batches=5)

import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device,
num_epochs=num_epochs, eval_freq=5, eval_iter=5,
start_context=format_input(val_data[0]), tokenizer=tokenizer)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


torch.manual_seed(123)
for entry in test_data[:3]: #1
    input_text = format_input(entry)
    token_ids = generate( #2
    model=model,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=256,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
    generated_text[len(input_text):]
    .replace("### Response:", "")
    .strip()
    )
    print(input_text)
    print(f"\nCorrect response:\\n>> {entry['output']}")
    print(f"\nModel response:\\n>> {response_text.strip()}")
    print("-------------------------------------")


from tqdm import tqdm
import json
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=256,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
    generated_text[len(input_text):]
    .replace("### Response:", "")
    .strip()
    )
    test_data[i]["model_response"] = response_text
    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)

import re
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth" #1
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")