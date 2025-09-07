# Qwen Model Fine-tuning for Human-Robot Interaction

This project uses LoRA (Low-Rank Adaptation) technology to perform parameter-efficient fine-tuning of Qwen2.5/Qwen3 models, specifically optimized for human-robot interaction dialogue capabilities. The project adopts a modular design with clear responsibilities, making it easy to maintain and extend.

## 🚀 Key Features

- **Parameter Efficient**: Uses LoRA technology, training only <1% of model parameters
- **Memory Friendly**: Supports 4bit/8bit quantization to reduce GPU memory requirements
- **Modular Design**: Complete separation of training, evaluation, and data processing functions
- **Multi-scenario Support**: Four major interaction scenarios: assembly, delivery, positioning, relocation
- **Experiment Tracking**: Supports WandB and TensorBoard
- **Easy to Extend**: Clean code structure for easy addition of new features

## 📁 Project Structure

```
disLLM/
├── 📂 data/                      # Raw data directory
│   ├── assembly.json             # Assembly scenario data
│   ├── delivery.json             # Delivery scenario data
│   ├── position.json             # Position scenario data
│   └── relocation.json           # Relocation scenario data
├── 📂 processed_data/            # Preprocessed data
│   ├── train.json                # Training data
│   └── validation.json           # Validation data
├── 📂 qwen_finetuned/           # Fine-tuned model output
├── 🔧 data_processor.py          # Data preprocessing module
├── 🚀 finetuning_trainer.py      # LoRA fine-tuning trainer (focused on training)
├── 📊 evaluator.py               # Model evaluator (focused on evaluation)
├── 🎯 train_finetune.py          # Main training script (workflow coordination)
├── 📋 requirements.txt           # Dependencies list
└── 📖 README.md                  # Project documentation
```

## 🏗️ Modular Architecture

### Core Modules

| Module | Function | Main Classes/Functions |
|--------|----------|------------------------|
| `data_processor.py` | Data preprocessing | `MultiwozDataProcessor` |
| `finetuning_trainer.py` | Model training | `QwenFineTuner`, `FineTuningConfig` |
| `evaluator.py` | Model evaluation | `ModelEvaluator` |
| `train_finetune.py` | Workflow coordination | `main()` |

### Architecture Advantages
- **Separation of Concerns**: Each module focuses on a single function
- **Easy Maintenance**: Modifying evaluation logic doesn't affect training code
- **Extensibility**: Can independently replace or extend any module
- **Reusability**: Evaluator can be used independently in other projects

## 📊 Dataset Description

The dataset contains dialogue data for 4 human-robot interaction scenarios in Multiwoz format:

- **assembly**: Mechanical product assembly tasks
- **delivery**: Package delivery tasks  
- **position**: Robot navigation and position management
- **relocation**: Object relocation tasks

### Data Format
```json
{
  "text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>",
  "domain": "assembly",
  "user_input": "Can you help me assemble a phone?",
  "system_response": "Sure, I can help with that. What color phone do you need?",
  "slots": "{...}"
}
```

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional)

```bash
# If using wandb for experiment tracking
export WANDB_PROJECT="qwen-finetuning"
export WANDB_API_KEY="your_api_key"

# Set HuggingFace cache directory
export HF_HOME="/path/to/your/cache"
```

## 🚀 Quick Start

### 1. Complete Fine-tuning Workflow

```bash
# Default configuration training (recommended for beginners)
python train_finetune.py --data_dir data --output_dir ./qwen_finetuned

# Skip data preprocessing (if already processed)
python train_finetune.py --skip_preprocessing --output_dir ./qwen_finetuned
```

### 2. Custom Parameter Fine-tuning

```bash
python train_finetune.py \
    --data_dir data \
    --output_dir ./my_qwen_model \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --max_length 2048
```

### 3. Using Different Base Models

```bash
# Use Qwen2.5-14B (requires more GPU memory)
python train_finetune.py --model_name Qwen/Qwen2.5-14B-Instruct

# Use Qwen2.5-32B (requires substantial GPU memory)
python train_finetune.py --model_name Qwen/Qwen2.5-32B-Instruct

# Use smaller model (when GPU memory is limited)
python train_finetune.py --model_name Qwen/Qwen2.5-1.5B-Instruct
```

### 4. Evaluate Trained Model Only

```bash
# Quick test (using train_finetune.py)
python train_finetune.py --evaluate_only --output_dir ./qwen_finetuned

# Detailed evaluation (using dedicated evaluator)
python evaluator.py --model_path ./qwen_finetuned --test_data processed_data/validation.json
```

## Training Parameters

### Basic Parameters
- `--model_name`: Base model name, supports Qwen2.5 and Qwen3 series
- `--epochs`: Number of training epochs, default 3
- `--batch_size`: Batch size, default 2
- `--learning_rate`: Learning rate, default 2e-4
- `--max_length`: Maximum sequence length, default 2048

### LoRA Parameters
- `--lora_r`: LoRA rank, controls adapter size, default 16
- `--lora_alpha`: LoRA scaling parameter, default 32
- `--lora_dropout`: LoRA dropout rate, default 0.1

### Quantization Parameters
- `--use_8bit`: Use 8bit quantization instead of 4bit
- Default uses 4bit quantization to save GPU memory

## Technical Features

### LoRA Fine-tuning Advantages
1. **Parameter Efficiency**: Train only a small number of adapter parameters (typically <1% of model parameters)
2. **Memory Friendly**: Significantly reduces GPU memory requirements during training
3. **Fast Training**: Training speed is several times faster than full parameter fine-tuning
4. **Modular**: Can train different adapters for different tasks

### Quantization Technology
- **4bit Quantization**: Uses QLoRA technology to drastically reduce memory usage while maintaining performance
- **Nested Quantization**: Further optimizes memory usage
- **Computation Precision**: Critical computations still use float16 to ensure accuracy

### Dialogue Format Optimization
- **System Prompts**: Customized system prompts for each domain
- **Qwen Format**: Uses standard Qwen dialogue templates
- **Context Awareness**: Preserves domain and slot information

## Model Performance

### Hardware Requirements
- **Minimum Configuration**: 16GB GPU memory (using 4bit quantization)
- **Recommended Configuration**: 24GB+ GPU memory
- **CPU**: Multi-core processor, 16+ cores recommended
- **Memory**: 32GB+ system memory

### Training Time Estimation
- **Qwen2.5-7B**: About 2-4 hours (single A100/H100)
- **Qwen2.5-14B**: About 4-8 hours
- **Qwen2.5-32B**: About 8-16 hours

## Evaluation Metrics

### 1. Domain Understanding Accuracy
Evaluates the model's understanding of different interaction scenarios

### 2. Slot Filling Accuracy  
Evaluates the model's ability to extract structured information

### 3. Response Quality Scores
- **Coherence**: Logic and fluency of responses
- **Relevance**: Degree of relevance between response and user input  
- **Helpfulness**: Usefulness of response to the user

## Experiment Tracking

### Using wandb (Recommended)
```bash
export WANDB_PROJECT="qwen-finetuning"
export WANDB_API_KEY="your_api_key"
python train_finetune.py
```

### Using tensorboard
```bash
tensorboard --logdir ./qwen_finetuned/logs
```

## 📊 Model Evaluation

### Quick Evaluation
```bash
# Use dedicated evaluator for detailed evaluation
python evaluator.py \
    --model_path ./qwen_finetuned \
    --test_data processed_data/validation.json \
    --output evaluation_results.json
```

### Evaluation Metrics Description
- **Domain Accuracy**: Understanding accuracy for each scenario
- **Slot Accuracy**: Structured information extraction accuracy  
- **Response Quality**: Coherence, relevance, and helpfulness scores

## 🔧 Model Inference

### Using Evaluator for Inference
```python
from evaluator import ModelEvaluator

# Load fine-tuned model
evaluator = ModelEvaluator("./qwen_finetuned", base_model_name="Qwen/Qwen2.5-7B-Instruct")

# Generate response
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nCan you help me assemble a phone?<|im_end|>\n<|im_start|>assistant\n"
response = evaluator.generate_response(prompt)
print(response)
```

### Batch Inference
```python
test_prompts = [
    "I need to deliver a package to the warehouse.",
    "Where can I find the laser cell?", 
    "Bring me the red fuse from the electrical box."
]

for user_input in test_prompts:
    # Build complete prompt
    full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    response = evaluator.generate_response(full_prompt)
    print(f"User: {user_input}")
    print(f"Assistant: {response}\n")
```

## Performance Optimization Recommendations

### 1. Memory Optimization
- Use gradient checkpointing: `gradient_checkpointing=True`
- Adjust batch size and gradient accumulation steps
- Use DeepSpeed ZeRO optimization (for large models)

### 2. Training Acceleration
- Use fused optimizer: `optim="adamw_torch_fused"`
- Enable mixed precision training (FP16)
- Use DataLoader optimization

### 3. LoRA Tuning
- **Increase lora_r**: Improve model capacity but increase parameter count
- **Adjust lora_alpha**: Control adapter influence strength
- **Select target modules**: Fine-tune specific layers

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Tensor Dimension Mismatch Error
```
ValueError: expected sequence of length 77 at dim 1 (got 49)
```
**Solution**: This issue has been fixed in the project using a custom data collator to handle sequences of different lengths.

#### 2. Out of Memory (OOM)
```bash
# Solution 1: Reduce batch size
python train_finetune.py --batch_size 1

# Solution 2: Use 8bit quantization
python train_finetune.py --use_8bit

# Solution 3: Use smaller model
python train_finetune.py --model_name Qwen/Qwen2.5-1.5B-Instruct

# Solution 4: Reduce maximum sequence length
python train_finetune.py --max_length 1024
```

#### 3. Slow Training Speed
```bash
# Increase batch size (if GPU memory allows)
python train_finetune.py --batch_size 4

# Mixed precision training (already enabled by default)
# Fused optimizer (already enabled by default)
```

#### 4. Poor Model Performance
```bash
# Increase LoRA parameters
python train_finetune.py --lora_r 32 --lora_alpha 64

# Increase training epochs
python train_finetune.py --epochs 5

# Adjust learning rate
python train_finetune.py --learning_rate 3e-4
```

#### 5. Model Loading Failure
```bash
# Check if model path is correct
ls -la ./qwen_finetuned/

# Use base model name for evaluation
python evaluator.py --model_path ./qwen_finetuned --test_data processed_data/validation.json
```

### 🐛 Debugging Tips

1. **Enable Verbose Logging**
   ```bash
   export TRANSFORMERS_VERBOSITY=debug
   python train_finetune.py
   ```

2. **Check Data Format**
   ```python
   import json
   with open('processed_data/train.json', 'r') as f:
       data = [json.loads(line) for line in f]
   print(data[0])  # Check first sample
   ```

3. **Monitor GPU Usage**
   ```bash
   # Open another terminal for monitoring
   watch -n 1 nvidia-smi
   ```

## Advanced Features

### 1. Multi-GPU Training
```bash
# Use DistributedDataParallel
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_finetune.py
```

### 2. Checkpoint Recovery
```bash
# Continue training from checkpoint
python train_finetune.py --resume_from_checkpoint ./qwen_finetuned/checkpoint-1000
```

### 3. Custom LoRA Configuration
```python
config = FineTuningConfig(
    lora_target_modules=["q_proj", "v_proj", "o_proj"],  # Custom target modules
    lora_r=64,  # Larger rank
    lora_alpha=128  # Adjust alpha accordingly
)
```

## License

This project is licensed under the MIT License.

## Contributing

We welcome contributions! Please feel free to submit Issues and Pull Requests to improve the project.

## Contact

If you have any questions, please contact chen Li through GitHub Issues.