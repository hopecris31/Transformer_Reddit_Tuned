import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead
import evaluate

##### FINE-TUNING A PRE-TRAINED MODEL #####
print("Fine-tuning GPT-2")

checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(checkpoint)

##### PREPARING THE DATA #####
print("Preparing the Data")
raw_datasets = load_dataset("reddit", split="train[:50%]")

print("Column Names: ", raw_datasets.column_names)

print("Tokenizing the Data")

def tokenize_function(example):
    return tokenizer(example["subreddit"], example["content"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Training")



def compute_metrics(eval_preds):
    metric = evaluate.load("reddit")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
                                output_dir="./redditGPT",  # The output directory
                                overwrite_output_dir=True,  # overwrite the content of the output directory
                                num_train_epochs=2,
                                per_device_train_batch_size=8,  # batch size for training
                                per_device_eval_batch_size=16,  # batch size for evaluation
                                eval_steps=200,  # Number of update steps between two evaluations.
                                save_steps=400,  # after # steps model is saved
                                warmup_steps=250,  # number of warmup steps for learning rate scheduler
                                prediction_loss_only=True,
                                evaluation_strategy="steps",
                                label_names=["content", "subreddit"]
                                )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

trainer.train()
path = "C:/Users/hopec/Desktop/NLP_Transformer"
trainer.save_model(path + "/ChatGRT")

trainer.evaluate(tokenized_datasets)