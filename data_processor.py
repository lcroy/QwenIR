#!/usr/bin/env python3
"""
数据预处理模块 - 将Multiwoz格式数据转换为适合微调的格式
"""

import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
import random

@dataclass
class DialogTurn:
    """对话轮次数据结构"""
    user_input: str
    system_response: str
    domain: str
    slots: Dict[str, Any]
    search_result: Dict[str, Any]

class MultiwozDataProcessor:
    """Multiwoz格式数据处理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.domains = ["assembly", "delivery", "position", "relocation"]
        
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_dialog_turns(self, data: Dict[str, Any]) -> List[DialogTurn]:
        """从原始数据中提取对话轮次"""
        turns = []
        
        for dialog_id, dialog_data in data.items():
            domain = list(dialog_data["domain"].keys())[0]
            
            for turn in dialog_data["turn"]:
                dialog_turn = DialogTurn(
                    user_input=turn["user"],
                    system_response=turn["system"],
                    domain=domain,
                    slots=turn.get("slots", {}),
                    search_result=turn.get("search_result", {})
                )
                turns.append(dialog_turn)
        
        return turns
    
    def create_training_prompts(self, turns: List[DialogTurn]) -> List[Dict[str, str]]:
        """创建训练提示词 - 专为微调优化"""
        training_data = []
        
        for turn in turns:
            # 为微调优化的格式：包含系统提示词
            system_prompt = f"You are a helpful assistant for {turn.domain} tasks. You help users with various requests related to {turn.domain}."
            
            # 用户输入
            user_message = turn.user_input
            
            # 助手回复
            assistant_message = turn.system_response
            
            # Qwen格式的对话模板
            conversation = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_message}<|im_end|>"
            )
            
            training_data.append({
                "text": conversation,
                "domain": turn.domain,
                "user_input": turn.user_input,
                "system_response": turn.system_response,
                "slots": json.dumps(turn.slots, ensure_ascii=False),
                "system_prompt": system_prompt
            })
        
        return training_data
    
    def process_all_data(self) -> DatasetDict:
        """处理所有数据文件"""
        all_training_data = []
        
        for domain in self.domains:
            file_path = os.path.join(self.data_dir, f"{domain}.json")
            if os.path.exists(file_path):
                print(f"Processing {domain} data...")
                data = self.load_json_file(file_path)
                turns = self.extract_dialog_turns(data)
                domain_data = self.create_training_prompts(turns)
                all_training_data.extend(domain_data)
                print(f"Processed {len(domain_data)} samples from {domain}")
        
        # 转换为HuggingFace Dataset格式
        dataset = Dataset.from_list(all_training_data)
        
        # 划分训练集和验证集
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        
        return DatasetDict({
            "train": train_test_split["train"],
            "validation": train_test_split["test"]
        })
    
    def save_processed_data(self, dataset: DatasetDict, output_dir: str = "processed_data"):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON格式
        for split in ["train", "validation"]:
            output_file = os.path.join(output_dir, f"{split}.json")
            dataset[split].to_json(output_file)
            print(f"Saved {split} data to {output_file}")
        
        # 保存统计信息
        stats = {
            "total_samples": len(dataset["train"]) + len(dataset["validation"]),
            "train_samples": len(dataset["train"]),
            "validation_samples": len(dataset["validation"]),
            "domains": self.domains
        }
        
        with open(os.path.join(output_dir, "stats.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Data processing completed. Total samples: {stats['total_samples']}")

def main():
    """主函数"""
    processor = MultiwozDataProcessor()
    dataset = processor.process_all_data()
    processor.save_processed_data(dataset)
    
    # 显示数据样例
    print("\n=== Sample Training Data ===")
    sample = dataset["train"][0]
    print(f"Domain: {sample['domain']}")
    print(f"Text: {sample['text'][:200]}...")

if __name__ == "__main__":
    main()
