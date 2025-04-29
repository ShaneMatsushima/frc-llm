from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch
from datasets import Dataset
import yaml

def load_base_model(model_name="gpt2-medium"):
    """Load a base GPT-2 model and tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add special tokens for FRC-specific concepts
    special_tokens = ['<alliance>', '<autonomous>', '<teleop>', '<endgame>', 
                     '<ranking_points>', '<penalty>', '<coopertition>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def train_model(train_data, model, tokenizer, config_path):
    """Fine-tune the model on FRC-specific data"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Prepare dataset
    train_dataset = Dataset.from_dict(train_data)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    
    # Set up training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        overwrite_output_dir=True,
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        save_steps=config['save_steps'],
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.save_model(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])
    
    return model