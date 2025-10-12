import os
import torch
import yaml
import warnings
import argparse

from peft import LoraConfig, TaskType, get_peft_model
from trainer.dataset.preprocess import JSONLDataset
from swift.llm import (
    get_model_tokenizer, get_template, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info
from swift.tuners import LoraConfig
from swift.trainers import TrainingArguments, Trainer

warnings.filterwarnings("ignore")
logger = get_logger()
data_seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# 读取配置
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to YAML config file")
parser.add_argument("--output_path", type=str, help="Path to save the output model")
parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                    help="Distributed launcher passes this argument automatically")

args, _ = parser.parse_known_args()
config_path = args.config
output_path = args.output_path
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 加载模型和处理器
model, processor = get_model_tokenizer(
    config["model"]["model_path"], 
    torch_dtype=torch.bfloat16, 
    attn_impl='flash_attn', 
    MIN_PIXELS=config["generate"]["min_pixels"], 
    MAX_PIXELS=config["generate"]["max_pixels"]
)

SYSTEM_MESSAGE = "You are a helpful assistant."

template = get_template(
    model.model_meta.template, 
    processor, 
    default_system=SYSTEM_MESSAGE, 
    max_length=config["generate"]["max_length"], 
    truncation_strategy='right', 
    max_pixels=config["generate"]["max_pixels"]
)
template.model = model
template.set_mode('train')

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config["lora"]["rank"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],
    lora_dtype='bfloat16',
    target_modules=list(config["lora"]["target_modules"]),
)

model = get_peft_model(model, lora_config)
logger.info(f'LoRA 配置: {lora_config}')
logger.info(f'加载完 LoRA 后模型结构: {model}')

# 查看模型可训练参数量
model_parameter_info = get_model_parameter_info(model)
logger.info(f'模型参数信息: {model_parameter_info}')

# 加载数据集
train_dataset = JSONLDataset(
    data_path=config["dataset"]["train_data_path"],
    jsonl_file_path=config["dataset"]["train_json_path"], 
    config_path=config_path, 
    format = "swift",
    attr_name='data-bbox'
)
eval_dataset = JSONLDataset(
    data_path=config["dataset"]["valid_data_path"],
    jsonl_file_path=config["dataset"]["valid_json_path"], 
    config_path=config_path, 
    format = "swift",
    attr_name='data-bbox'
)
train_dataset = LazyLLMDataset(train_dataset, template.encode, random_state=data_seed)
eval_dataset = LazyLLMDataset(eval_dataset, template.encode, random_state=data_seed)

training_args = TrainingArguments(
    num_train_epochs=config["hparams"]["num_train_epochs"],
    logging_steps=config["hparams"]["log_every_steps"],
    output_dir=output_path,
    eval_strategy="steps",
    report_to=['tensorboard'],
    optim=config["hparams"]["optim"],
    eval_steps=config["hparams"]["eval_every_steps"],
    learning_rate=float(config["hparams"]["learning_rate"]),
    per_device_train_batch_size=config["hparams"]["batch_size"],
    per_device_eval_batch_size=config["hparams"]["eval_batch_size"] or config["hparams"]["batch_size"],
    gradient_checkpointing=config["hparams"]["gradient_checkpointing"],
    gradient_checkpointing_kwargs=(
        dict(use_reentrant=False) 
        if config["hparams"]["gradient_checkpointing"] and config["lora"] is not None
        else {}
    ),
    gradient_accumulation_steps=config["hparams"]["gradient_accumulation_steps"],
    weight_decay=config["hparams"]["weight_decay"],
    dataloader_num_workers=config["max_workers"],
    load_best_model_at_end=True,
    save_strategy="steps",
    ddp_find_unused_parameters=config["hparams"]["find_unused_parameters"],
    save_steps=config["hparams"]["save_every_steps"],
    warmup_steps=config["hparams"]["warmup_steps"],
    warmup_ratio=config["hparams"]["warmup_ratio"],
    bf16=True,
    label_names=["labels"],  
    max_grad_norm=config["hparams"]["clip_grad_norm"],
    remove_unused_columns=False,
    eval_on_start=False,
    metric_for_best_model=config["dataset"]["metric_for_best_model"],
)

model.enable_input_require_grads()

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator, 
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    template=template,
)

trainer.train()
