# Major change since July 9 for scaling up
# Major change since July 18 for fixing the lora bug
import transformers
from transformers import LlamaTokenizer
import os
import torch
import torch.nn as nn
import random
from dataclasses import dataclass, field
from typing import Optional
from icae.utils import stable_trainer
from peft import (
    get_peft_model,
)
from transformers.trainer_utils import get_last_checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dataset, eval_dataset, training_args, data_collator=None):

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    trainer = stable_trainer.StableTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    trainer.accelerator.dispatch_batches = False    # To avoid the padding error introduced in transformers==4.31.0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
def instruct_ft_tokenize_function(examples, model, mem):
    text_output = model.tokenizer(examples["input"], truncation=True, max_length=model.training_args.model_max_length, padding=False, return_attention_mask=False)
    prompt_output = model.tokenizer(examples['prompt'], add_special_tokens=False, padding=False)
    answer_output = model.tokenizer(examples['answer'], add_special_tokens=False, padding=False)

    text_output['prompt_answer_ids'] = []
    text_output['labels'] = []
    for idx in range(len(text_output["input_ids"])):
        # add mem tokens to the input
        text_output["input_ids"][idx] += mem

        # decoder part:
        prompt_ids = [model.ft_token_id] + prompt_output['input_ids'][idx] + [model.ft_token_id]
        answer_ids = answer_output['input_ids'][idx] + [model.eos_id]
        text_output['prompt_answer_ids'].append(prompt_ids + answer_ids)
        labels = [-100] * len(prompt_ids) + answer_ids
        text_output['labels'].append(labels)
    return text_output