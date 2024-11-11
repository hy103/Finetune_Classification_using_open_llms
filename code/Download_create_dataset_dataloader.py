import json
import os
import urllib
import torch
from torch.utils.data import Dataset, DataLoader

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            print(response)
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding = "utf-8") as file:
            file.write(text_data)

    else: 
        with open(file_path, "r", encoding = "utf-8") as file:
            text_data = file.read()
        with open(file_path, "r") as file:
            data = json.load(file)
        return data

file_path = "instruction-data.json"
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json")

data = download_and_load_file(file_path, url)



def format_input(entry):
    # print("-------------\n")
    # print(f"-------   Instruction : ------------- {entry}")
    # print("-------------\n")
    instruction_text = (
    f"Below is an instruction that describes a task. "
    f"Write a response that appropriately completes the request."
    f"\n\\n### Instruction:\\n{entry['instruction']}"
    )
    input_text = (
    f"\n\\n### Input:\\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

train_portion = int(len(data)*0.8)
test_portion = int(len(data)*0.1)
val_portion = len(data) - train_portion - test_portion 

train_data = data[:train_portion]
test_data = data[train_portion: train_portion + test_portion]
val_data = data[train_portion + test_portion:]  # Ensure this is a list of dictionaries



class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_text = []

        for entry in data :
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n ### Response: \n {entry['output']}"
            full_text = instruction_plus_input+response_text
            self.encoded_text.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_text[index]
    
    def __len__(self):
        return len(self.data)
    

def custom_collate_fn(
        batch, 
        pad_token_id = 50256,
        ignore_index = -100,
        allowed_max_length = None,
        device = "cpu"
):
    batch_max_lenght = max(len(item) for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        padded = (
            new_item+([pad_token_id]*(batch_max_lenght - len(new_item)))
        )
        inputs = torch.tensor(padded)
        targets = torch.tensor(padded[1:] + [pad_token_id])
        
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
   
        ## If condition checks if there more than 1 non zero index indicating
        ## mpore th
        if indices.numel()>1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(
    custom_collate_fn,
    device = device,
    allowed_max_length = 1024
)


from torch.utils.data import DataLoader

num_workers =0
batch_size = 8

torch.manual_seed(123)
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    num_workers=num_workers,
    shuffle= True,
    collate_fn=customized_collate_fn,
    drop_last=True
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    num_workers=num_workers,
    shuffle= True,
    collate_fn=customized_collate_fn,
    drop_last=True
)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    num_workers=num_workers,
    shuffle= True,
    collate_fn=customized_collate_fn,
    drop_last=True
)

# Define paths for saving datasets and tokenizer
torch.save({
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "test_dataset": test_dataset,
    "train_data": train_data,
    "val_data": val_data,
    "test_data": test_data,
    "tokenizer": tokenizer
}, "datasets_and_tokenizer.pth")

# Save collate function settings if they are customized
torch.save({
    "device": device,
    "allowed_max_length": 1024,
    "pad_token_id": 50256,
    "ignore_index": -100
}, "collate_fn_settings.pth")

