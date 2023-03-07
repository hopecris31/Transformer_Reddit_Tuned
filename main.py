import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import DataCollatorWithPadding

from transformers import TrainingArguments
from transformers import Trainer
import evaluate

checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

raw_datasets = load_dataset("reddit")
raw_datasets

def tokenize_function(example):
  #NEED TO FILL THIS IN
  return ""

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

example_inputs = ["Your name is ", "How do I ", "You drop into the world, and there lies a "]


