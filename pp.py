# Install necessary libraries
!pip install -U transformers datasets tensorboard sentencepiece accelerate evaluate rouge_score

import torch
import pprint
import evaluate
import numpy as np
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# Pretty print for better output readability
pretty_printer = pprint.PrettyPrinter()

# Load dataset and split into training and validation sets
data = load_dataset('bbc-news-summary', split='train')
split_data = data.train_test_split(test_size=0.3, shuffle=True)  # Changed split ratio
train_set = split_data['train']
validation_set = split_data['test']
print(train_set)
print(validation_set)

# Model parameters
MODEL_CHECKPOINT = 't5-small' 
BATCH_SZ = 8  
WORKER_COUNT = 2  
EPOCH_COUNT = 5 
RESULTS_DIR = 't5_small_results'
MAX_INPUT_LENGTH = 256 

tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)

# Function for preprocessing data
def process_batch(batch):
    input_texts = [f"abstract: {doc}" for doc in batch['Articles']]  # Changed prefix
    tokenized_inputs = tokenizer(input_texts, max_length=MAX_INPUT_LENGTH, truncation=True, padding='max_length')

    target_texts = batch['Summaries']
    with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(target_texts, max_length=MAX_INPUT_LENGTH, truncation=True, padding='max_length')

    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    return tokenized_inputs

# Tokenize datasets
train_tokenized = train_set.map(process_batch, batched=True, num_proc=WORKER_COUNT)
valid_tokenized = validation_set.map(process_batch, batched=True, num_proc=WORKER_COUNT)

# Load model
summarizer_model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer_model.to(device_type)

# Print parameter counts
total_model_params = sum(param.numel() for param in summarizer_model.parameters())
trainable_model_params = sum(param.numel() for param in summarizer_model.parameters() if param.requires_grad)

print(f"Total parameters: {total_model_params:,}")
print(f"Trainable parameters: {trainable_model_params:,}")

# Load evaluation metric
metric_rouge = evaluate.load("rouge")

# Metric computation function
def calculate_metrics(predictions_with_labels):
    predictions, references = predictions_with_labels.predictions[0], predictions_with_labels.label_ids

    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    references = np.where(references != -100, references, tokenizer.pad_token_id)
    decoded_references = tokenizer.batch_decode(references, skip_special_tokens=True)

    results = metric_rouge.compute(
        predictions=decoded_predictions,
        references=decoded_references,
        use_stemmer=True,
        rouge_types=['rouge1', 'rouge2', 'rougeL']
    )

    lengths = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    results["average_length"] = np.mean(lengths)

    return {key: round(val, 4) for key, val in results.items()}

# Helper function to process logits
def logits_processor(logits, references):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, references

# Training arguments
training_parameters = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=EPOCH_COUNT,
    per_device_train_batch_size=BATCH_SZ,
    per_device_eval_batch_size=BATCH_SZ,
    warmup_steps=200,  
    weight_decay=0.02,  
    logging_dir=RESULTS_DIR,
    logging_steps=20,  
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1, 
    learning_rate=5e-5,  
    dataloader_num_workers=WORKER_COUNT
)

# Initialize Trainer
trainer_instance = Trainer(
    model=summarizer_model,
    args=training_parameters,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    compute_metrics=calculate_metrics,
    preprocess_logits_for_metrics=logits_processor
)

# Train the model
trainer_instance.train()

# Save the tokenizer and model
tokenizer.save_pretrained(RESULTS_DIR)

# Summarization function
def summarize_document(text, model, tokenizer, max_tokens=256, beams=3):  
    input_tokens = tokenizer.encode(
        "abstract: " + text,
        return_tensors='pt',
        max_length=max_tokens,
        truncation=True
    ).to(device_type)

    summary_ids = model.generate(input_tokens, max_length=60, num_beams=beams)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
