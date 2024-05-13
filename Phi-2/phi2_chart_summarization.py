from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch
from datasets import load_dataset
from trl import SFTTrainer

import pandas as pd
from datasets import Dataset, DatasetDict

base_model = "microsoft/phi-2"
new_model = "phi-2-chart_summarization"


train_data = pd.read_csv("./train_phi2.csv")
test_data = pd.read_csv("./test.csv")
valid_data = pd.read_csv("./valid.csv")

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
valid_dataset = Dataset.from_pandas(valid_data)

dataset_dict = DatasetDict({
    'train': train_dataset,
        'validation': valid_dataset,
        'test' : test_dataset
})

# Load base model(Phi-2)
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map=0,
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense',
        'fc1',
        'fc2',
    ]
)
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="./phi-2-chartSummarization",
    num_train_epochs=200,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_steps=100,
    logging_strategy="steps",
    learning_rate=2e-4,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    group_by_length=True,
    disable_tqdm=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_dict['train'],
    peft_config=peft_config,
    max_seq_length= 2048,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()

trainer.model.save_pretrained(new_model)

