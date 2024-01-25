import argparse
from tqdm import tqdm
import torch
import copy
from typing import List
import os
import sys
import transformers
import numpy as np
import jsonlines

import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.functional import max_pool1d, avg_pool1d
from einops import rearrange
from sklearn.cluster import KMeans, MiniBatchKMeans
from trainer import MemTrainer, MemDataCollatorForSeq2Seq

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def train(
    base_model: str,
    data_path: str = "data",
    output_dir: str = "/dccstor/jerryli1/mem",
    # training hyperparams
    batch_size: int = 16,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    resume_from_checkpoint: str = None,
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        load_in_4bit=True,
        device_map=device_map,
        use_auth_token="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",  # need auth token
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.eos_id = 1
    model.ft_token_id = model.config.vocab_size + 1

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def instruct_ft_tokenize_function(examples):
        text_output = tokenizer(examples["input"], truncation=True, max_length=512, padding=False, return_attention_mask=False)
        prompt_output = tokenizer(examples['prompt'], add_special_tokens=False, padding=False)
        answer_output = tokenizer(examples['answer'], add_special_tokens=False, padding=False)

        text_output['prompt_answer_ids'] = []
        text_output['labels'] = []
        # print(text_output["input_ids"])
        for idx in range(len(text_output["input_ids"])):
            # decoder part:
            prompt_ids = [model.ft_token_id] + prompt_output['input_ids'][idx] + [model.ft_token_id]
            answer_ids = answer_output['input_ids'][idx] + [model.eos_id]
            text_output['prompt_answer_ids'].append(prompt_ids + answer_ids)
            labels = [-100] * len(prompt_ids) + answer_ids
            text_output['labels'].append(labels)
        return text_output

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    if lora_target_modules == "all":
        print("Using all modules for LoRA")
        def find_all_linear_names(model):
            cls = bnb.nn.Linear4bit
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names:
                lora_module_names.remove('lm_head')
            return list(lora_module_names)

        lora_target_modules = find_all_linear_names(model)
        print("Using modules:", lora_target_modules)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    data = load_dataset(data_path, split='train[:100]')

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(instruct_ft_tokenize_function, batched=True)
        )
        val_data = (
            train_val["test"].shuffle().map(instruct_ft_tokenize_function, batched=True)
        )
    else:
        train_data = data.shuffle().map(instruct_ft_tokenize_function, batched=True)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    train_data.set_format(type="torch", columns=["input_ids", "prompt_answer_ids", "labels"])
    print(train_data[0])

    trainer = MemTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="paged_adamw_32bit",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=None,
            remove_unused_columns=False,
        ),
        data_collator=MemDataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

def random_removal(past_key_values, compress_to=200):  # (batch_size, num_heads, sequence_length, embed_size_per_head)
    compressed_past_kv = []
    for j in range(len(past_key_values)):
        rand_indices = np.random.choice(past_key_values[j][0].shape[2], size=compress_to, replace=False)
        compressed_past_kv.append([])
        for k in range(len(past_key_values[j])):
            compressed = past_key_values[j][k][:, :, rand_indices, :]
            compressed_past_kv[j].append(compressed)
        compressed_past_kv[j] = tuple(compressed_past_kv[j])
    compressed_past_kv = tuple(compressed_past_kv)

    return compressed_past_kv

def max_pool(past_key_values, compress_to=200):
    compressed_past_kv = []
    for j in range(len(past_key_values)):
        compressed_past_kv.append([])
        for k in range(len(past_key_values[j])):
            b, n, s, e = past_key_values[j][k].shape
            compressed = rearrange(max_pool1d(rearrange(past_key_values[j][k], 'b n s e -> b (n e) s'), kernel_size=2), 'b (n e) s -> b n s e', b=b, n=n, e=e)
            compressed_past_kv[j].append(compressed)
        compressed_past_kv[j] = tuple(compressed_past_kv[j])
    compressed_past_kv = tuple(compressed_past_kv)

    return compressed_past_kv

def kmeans(past_key_values, compress_to=200):
    compressed_past_kv = []
    for j in range(len(past_key_values)):
        joined_kv = torch.stack([past_key_values[j][0], past_key_values[j][1]])
        z, b, n, s, e = joined_kv.shape
        centers = MiniBatchKMeans(n_clusters=compress_to, random_state=0, n_init='auto').fit(rearrange(joined_kv, 'z b n s e -> s (z b n e)').detach().cpu().numpy()).cluster_centers_
        compressed = rearrange(torch.from_numpy(centers).to(device), 's (z b n e) -> z b n s e', z=z, b=b, n=n, e=e)
        compressed_past_kv.append((compressed[0], compressed[1]))
    compressed_past_kv = tuple(compressed_past_kv)

    return compressed_past_kv

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--llm', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--compress_to', type=int, default=200)
    parser.add_argument('--compress_method', type=str, default='random_removal')

    args = parser.parse_args()
    print(args)

    pwc_test = load_dataset('data', split='test[:10]')
    print(pwc_test)

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, use_auth_token=True, device_map='auto')

    def tokenize(example):
        text_output = tokenizer(example['input'], padding=False, return_attention_mask=False)
        prompt_output = tokenizer('Question: ' + example['prompt'] + '\n\nAnswer: ', add_special_tokens=False, padding=False, return_attention_mask=False)
        
        text_output['prompt_ids'] = prompt_output['input_ids']
        return text_output

    pwc_test = pwc_test.map(tokenize)
    pwc_test.set_format(type='torch')

    test_dataloader = DataLoader(
        pwc_test,
        batch_size=1,
    )

    outputs = []
    for batch in tqdm(test_dataloader):
        # print(batch)
        input_ids = batch['input_ids'].to(device)
        prompt_ids = batch['prompt_ids'].to(device)

        with torch.no_grad():
            out = model.forward(input_ids=input_ids, use_cache=True)

            past_key_values = out.past_key_values

            if args.compress_method == 'random_removal':
                compressed_past_kv = random_removal(past_key_values, compress_to=args.compress_to)
            elif args.compress_method == 'max_pool':
                compressed_past_kv = max_pool(past_key_values, compress_to=args.compress_to)
            elif args.compress_method == 'kmeans':
                compressed_past_kv = kmeans(past_key_values, compress_to=args.compress_to)
            
            compressed_length = compressed_past_kv[0][0].shape[2]
            compressed_output = model.generate(input_ids=torch.cat([torch.ones_like(input_ids[:, -compressed_length:]), prompt_ids], dim=1), past_key_values=compressed_past_kv, max_new_tokens=100)
            compressed_decoded = tokenizer.batch_decode(compressed_output, skip_special_tokens=True)
            
            orig_output = model.generate(torch.cat([input_ids, prompt_ids], dim=1), max_new_tokens=100)
            orig_decoded = tokenizer.batch_decode(orig_output, skip_special_tokens=True)

            for i, p, c, o in zip(batch['input'], batch['prompt'], compressed_decoded, orig_decoded):
                outputs.append({'input': i, 'prompt': p, 'compressed_answer': c.split('Answer: ')[1].strip(), 'uncompressed_answer': o.split('Answer: ')[1].strip()})

    with jsonlines.open('output.jsonl', 'w') as writer:
        writer.write_all(outputs)

if __name__ == "__main__":
    main()
    # train('meta-llama/Llama-2-7b-chat-hf')