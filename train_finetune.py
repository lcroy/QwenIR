#!/usr/bin/env python3
"""
Qwen模型微调主训练脚本
"""

import os
import argparse
import json
from data_processor import MultiwozDataProcessor
from finetuning_trainer import QwenFineTuner, FineTuningConfig
from evaluator import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model for human-robot interaction")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="./qwen_finetuned", help="Output directory for the model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Base model name (Qwen2.5 or Qwen3)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization instead of 4-bit")
    
    args = parser.parse_args()
    
    # 设置wandb环境变量（如果已配置API key）
    if os.getenv("WANDB_API_KEY") and not os.getenv("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = "qwen-finetuning"
        print("📊 WandB project set to 'qwen-finetuning'")
    
    print("=" * 60)
    print("🤖 Qwen Model Fine-tuning for Human-Robot Interaction")
    print("=" * 60)
    print(f"📂 Data directory: {args.data_dir}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"🔥 Base model: {args.model_name}")
    print(f"🔄 Training epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🎯 LoRA rank: {args.lora_r}")
    print(f"⚡ LoRA alpha: {args.lora_alpha}")
    print(f"📏 Max length: {args.max_length}")
    print(f"🔧 Quantization: {'8-bit' if args.use_8bit else '4-bit'}")
    
    # 显示wandb状态
    if os.getenv("WANDB_PROJECT"):
        print(f"📊 WandB tracking: ✅ (Project: {os.getenv('WANDB_PROJECT')})")
    else:
        print("📊 WandB tracking: ❌ (Set WANDB_API_KEY to enable)")
    
    print("=" * 60)
    
    # 步骤1: 数据预处理
    if not args.skip_preprocessing and not args.evaluate_only:
        print("\n=== Step 1: Data Preprocessing ===")
        processor = MultiwozDataProcessor(args.data_dir)
        dataset = processor.process_all_data()
        processor.save_processed_data(dataset)
        print("Data preprocessing completed!")
    
    # 步骤2: 模型微调
    if not args.evaluate_only:
        print("\n=== Step 2: Model Fine-tuning ===")
        
        # 配置微调参数
        config = FineTuningConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            max_length=args.max_length,
            use_4bit=not args.use_8bit  # 如果指定8bit，则不使用4bit
        )
        
        # 创建训练器并开始训练
        trainer = QwenFineTuner(config)
        trainer.train(
            train_data_path="processed_data/train.json",
            eval_data_path="processed_data/validation.json"
        )
        print("Model fine-tuning completed!")
    
    # 步骤3: 模型评估
    if os.path.exists(args.output_dir):
        print("\n=== Step 3: Model Evaluation ===")
        
        # 使用专门的评估器
        evaluator = ModelEvaluator(args.output_dir, base_model_name=args.model_name)
        
        # 简单测试生成
        print("\n=== Quick Generation Test ===")
        test_prompts = [
            "Can you help me assemble a phone?",
            "I need to deliver a package to the warehouse.",
            "Where can I find the laser cell?",
            "Bring me the red fuse from the electrical box."
        ]
        
        for prompt in test_prompts:
            print(f"\nUser: {prompt}")
            # 构建完整的prompt
            full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            response = evaluator.generate_response(full_prompt)
            print(f"Assistant: {response}")
    
    else:
        print(f"Model directory {args.output_dir} not found. Please train the model first.")
    
    print("\n=== Fine-tuning Pipeline Completed ===")

if __name__ == "__main__":
    main()
