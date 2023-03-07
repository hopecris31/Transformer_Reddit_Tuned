import torch
import numpy as np
import evaluate
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding


checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# download the reddit dataset from Huggingface
raw_datasets = load_dataset("reddit")


# split the datasets into a training(70%), validation(15%), and test(15%) sets.
train_and_val_data, test_data = train_test_split(raw_datasets, test_size=0.15, random_state=42)
train_data, val_data = train_test_split(raw_datasets, test_size=0.1765, random_state=42)

train_data.save_to_disk("train_dataset")
val_data.save_to_disk("val_dataset")
test_data.save_to_disk("test_dataset")

def tokenizer(string):
  tokenized_input = tokenizer(string, return_tensors="pt")
  return {"content": tokenized_input["content"][0]}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

example_inputs = ["Your name is ", "How do I do"]