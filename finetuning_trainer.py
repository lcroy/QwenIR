#!/usr/bin/env python3
"""
Qwen模型微调训练器 - 使用LoRA进行参数高效微调
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, TrainerCallback
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import json
import os
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import wandb
from tqdm import tqdm
import time

@dataclass
class FineTuningConfig:
    """微调训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # 可选择Qwen2.5或Qwen3
    
    # 训练配置
    output_dir: str = "./finetuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # 量化配置
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # 其他配置
    max_length: int = 2048
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    seed: int = 42
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Qwen2/Qwen3的默认LoRA目标模块
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

class ProgressCallback(TrainerCallback):
    """自定义训练进度回调"""
    
    def __init__(self):
        self.start_time = None
        self.best_eval_loss = float('inf')
        
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时的回调"""
        self.start_time = time.time()
        print("🚀 Training started!")
        print(f"📊 Total training steps: {state.max_steps}")
        print(f"📈 Training epochs: {args.num_train_epochs}")
        print(f"🔧 Batch size: {args.per_device_train_batch_size}")
        print(f"📚 Learning rate: {args.learning_rate}")
        print("-" * 50)
    
    def on_log(self, args, state, control, model, logs=None, **kwargs):
        """每次记录日志时的回调"""
        if logs:
            current_step = state.global_step
            max_steps = state.max_steps
            progress_pct = (current_step / max_steps) * 100
            
            # 计算剩余时间
            elapsed_time = time.time() - self.start_time
            if current_step > 0:
                time_per_step = elapsed_time / current_step
                remaining_steps = max_steps - current_step
                eta_seconds = remaining_steps * time_per_step
                eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
            else:
                eta_str = "Unknown"
            
            # 打印训练进度
            if 'loss' in logs:
                print(f"📝 Step {current_step:4d}/{max_steps} ({progress_pct:5.1f}%) | "
                      f"Loss: {logs['loss']:.4f} | "
                      f"LR: {logs.get('learning_rate', 0):.2e} | "
                      f"ETA: {eta_str}")
            
            # 打印评估结果
            if 'eval_loss' in logs:
                eval_loss = logs['eval_loss']
                print(f"🎯 Evaluation | Loss: {eval_loss:.4f}", end="")
                
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    print(" ⭐ (New Best!)")
                else:
                    print()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时的回调"""
        current_epoch = int(state.epoch)
        total_epochs = int(args.num_train_epochs)
        print(f"✅ Epoch {current_epoch}/{total_epochs} completed")
        print("-" * 30)
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时的回调"""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print("🎉 Training completed!")
        print(f"⏱️  Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"🏆 Best eval loss: {self.best_eval_loss:.4f}")
        print("=" * 50)

class QwenFineTuner:
    """Qwen模型微调器"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 初始化模型和分词器
        self._setup_model()
        
    def _setup_model(self):
        """设置模型和分词器"""
        print(f"Loading model: {self.config.model_name}")
        
        # 量化配置
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        else:
            bnb_config = None
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 准备模型用于k-bit训练
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        self.model.print_trainable_parameters()
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """准备训练数据集"""
        print(f"Loading dataset from {data_path}")
        
        # 读取JSONL格式文件
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        # 使用分词器进行tokenization
        def tokenize_function(examples):
            # 对文本进行tokenization
            inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # 在数据收集器中处理padding，而不是在这里
                max_length=self.config.max_length,
                return_tensors=None,
                add_special_tokens=True
            )
            
            # 确保input_ids和labels都是正确的格式
            model_inputs = {}
            model_inputs["input_ids"] = inputs["input_ids"]
            model_inputs["attention_mask"] = inputs["attention_mask"]
            
            # 对于causal language modeling，labels就是input_ids的副本
            # 确保labels是正确的列表格式，而不是嵌套列表
            model_inputs["labels"] = [ids.copy() for ids in inputs["input_ids"]]
            
            return model_inputs
        
        dataset = Dataset.from_list(data)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, train_data_path: str, eval_data_path: str):
        """执行微调训练"""
        print("🔄 Preparing for fine-tuning training...")
        
        # 初始化wandb（如果配置了）
        if os.getenv("WANDB_PROJECT"):
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "qwen-finetuning"),
                name=f"qwen-lora-{self.config.lora_r}-{self.config.lora_alpha}",
                config={
                    "model_name": self.config.model_name,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.per_device_train_batch_size,
                    "epochs": self.config.num_train_epochs,
                    "max_length": self.config.max_length,
                }
            )
            print("📊 WandB initialized for experiment tracking")
        
        # 准备数据
        print("📚 Loading and tokenizing datasets...")
        train_dataset = self.prepare_dataset(train_data_path)
        eval_dataset = self.prepare_dataset(eval_data_path)
        
        print(f"📈 Training samples: {len(train_dataset)}")
        print(f"📉 Validation samples: {len(eval_dataset)}")
        
        # 数据整理器 - 自定义数据收集器来处理padding
        def custom_data_collator(features):
            # 获取最大长度
            max_length = max(len(f["input_ids"]) for f in features)
            
            # 创建batch
            batch = {}
            batch["input_ids"] = []
            batch["attention_mask"] = []
            batch["labels"] = []
            
            for f in features:
                input_ids = f["input_ids"]
                attention_mask = f["attention_mask"]
                labels = f["labels"]
                
                # 计算需要padding的长度
                padding_length = max_length - len(input_ids)
                
                # 添加padding
                padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                padded_attention_mask = attention_mask + [0] * padding_length
                padded_labels = labels + [-100] * padding_length  # -100会被忽略在loss计算中
                
                batch["input_ids"].append(padded_input_ids)
                batch["attention_mask"].append(padded_attention_mask)
                batch["labels"].append(padded_labels)
            
            # 转换为张量
            import torch
            batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
            
            return batch
        
        data_collator = custom_data_collator
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if os.getenv("WANDB_PROJECT") else None,
            seed=self.config.seed,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            disable_tqdm=False,  # 保持tqdm进度条
        )
        
        # 创建进度回调
        progress_callback = ProgressCallback()
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[progress_callback]
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        print("💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # 完成wandb记录
        if os.getenv("WANDB_PROJECT"):
            wandb.finish()
        
        print(f"🎯 Training completed! Model saved to {self.config.output_dir}")
    

def main():
    """主函数"""
    # 配置微调参数
    config = FineTuningConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir="./qwen_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        lora_r=16,
        lora_alpha=32
    )
    
    # 初始化训练器
    trainer = QwenFineTuner(config)
    
    # 开始训练
    trainer.train(
        train_data_path="processed_data/train.json",
        eval_data_path="processed_data/validation.json"
    )

if __name__ == "__main__":
    main()
