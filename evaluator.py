#!/usr/bin/env python3
"""
模型评估器 - 评估微调后模型的性能
"""

import torch
import json
import os
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from tqdm import tqdm

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, base_model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.base_model_name = base_model_name
        
        # 加载模型和分词器
        self._load_model()
        
    def _load_model(self):
        """加载微调后的模型"""
        # 检查是否是LoRA微调模型
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            print("Loading LoRA fine-tuned model...")
            # 加载base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            # 加载LoRA适配器
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            print("Loading standard fine-tuned model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """生成回复"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()
    
    def evaluate_domain_understanding(self, test_data: List[Dict]) -> Dict[str, float]:
        """评估领域理解能力"""
        domain_accuracy = {}
        domain_samples = {}
        
        for sample in test_data:
            domain = sample['domain']
            if domain not in domain_samples:
                domain_samples[domain] = []
            domain_samples[domain].append(sample)
        
        for domain, samples in domain_samples.items():
            correct = 0
            total = len(samples)
            
            for sample in tqdm(samples, desc=f"Evaluating {domain}"):
                user_input = sample['user_input']
                expected_response = sample['system_response']
                
                # 构建提示词（包含系统提示）
                system_prompt = f"You are a helpful assistant for {domain} tasks."
                prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
                
                # 生成回复
                generated_response = self.generate_response(prompt)
                
                # 简单的相似度评估（可以改进为更复杂的评估）
                if self._calculate_similarity(generated_response, expected_response) > 0.3:
                    correct += 1
            
            domain_accuracy[domain] = correct / total if total > 0 else 0.0
        
        return domain_accuracy
    
    def evaluate_slot_filling(self, test_data: List[Dict]) -> Dict[str, float]:
        """评估槽位填充能力"""
        slot_metrics = {}
        
        for sample in test_data:
            domain = sample['domain']
            if domain not in slot_metrics:
                slot_metrics[domain] = {'correct': 0, 'total': 0}
            
            user_input = sample['user_input']
            expected_slots = json.loads(sample['slots'])
            
            # 构建提示词
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            
            # 生成回复
            generated_response = self.generate_response(prompt)
            
            # 检查是否包含关键槽位信息
            slot_found = self._check_slots_in_response(generated_response, expected_slots, domain)
            
            if slot_found:
                slot_metrics[domain]['correct'] += 1
            slot_metrics[domain]['total'] += 1
        
        # 计算准确率
        slot_accuracy = {}
        for domain, metrics in slot_metrics.items():
            slot_accuracy[domain] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0.0
        
        return slot_accuracy
    
    def evaluate_response_quality(self, test_data: List[Dict]) -> Dict[str, float]:
        """评估回复质量"""
        quality_scores = {
            'coherence': [],
            'relevance': [],
            'helpfulness': []
        }
        
        for sample in tqdm(test_data, desc="Evaluating response quality"):
            user_input = sample['user_input']
            expected_response = sample['system_response']
            
            # 构建提示词
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            
            # 生成回复
            generated_response = self.generate_response(prompt)
            
            # 评估质量指标
            coherence = self._evaluate_coherence(generated_response)
            relevance = self._evaluate_relevance(generated_response, user_input)
            helpfulness = self._evaluate_helpfulness(generated_response, expected_response)
            
            quality_scores['coherence'].append(coherence)
            quality_scores['relevance'].append(relevance)
            quality_scores['helpfulness'].append(helpfulness)
        
        # 计算平均分数
        avg_scores = {}
        for metric, scores in quality_scores.items():
            avg_scores[metric] = np.mean(scores)
        
        return avg_scores
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单的词汇重叠度）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _check_slots_in_response(self, response: str, expected_slots: Dict, domain: str) -> bool:
        """检查回复中是否包含预期的槽位信息"""
        response_lower = response.lower()
        
        # 根据领域检查关键槽位
        if domain == "assembly":
            key_slots = ["product", "quantity", "producttype"]
        elif domain == "delivery":
            key_slots = ["object", "area", "location"]
        elif domain == "position":
            key_slots = ["position_name", "operation"]
        elif domain == "relocation":
            key_slots = ["object_name"]
        else:
            return False
        
        # 检查是否包含关键槽位信息
        found_slots = 0
        total_expected_slots = 0
        
        # 从嵌套结构中提取槽位值
        if domain in expected_slots:
            domain_data = expected_slots[domain]
            
            # 检查T_inform中的req字段
            if "T_inform" in domain_data and "req" in domain_data["T_inform"]:
                req_data = domain_data["T_inform"]["req"]
                for slot in key_slots:
                    if slot in req_data:
                        total_expected_slots += 1
                        slot_value = req_data[slot]
                        if isinstance(slot_value, str) and slot_value.lower() != "not_mentioned":
                            if slot_value.lower() in response_lower:
                                found_slots += 1
            
            # 检查DB_request中的req字段
            if "DB_request" in domain_data and "req" in domain_data["DB_request"]:
                req_data = domain_data["DB_request"]["req"]
                for slot in key_slots:
                    if slot in req_data:
                        total_expected_slots += 1
                        slot_value = req_data[slot]
                        if isinstance(slot_value, str) and slot_value.lower() != "not_mentioned":
                            if slot_value.lower() in response_lower:
                                found_slots += 1
        
        # 至少需要找到50%的槽位
        if total_expected_slots == 0:
            return False
        
        accuracy = found_slots / total_expected_slots
        return accuracy >= 0.5
    
    def _evaluate_coherence(self, response: str) -> float:
        """评估回复的连贯性"""
        # 简单的连贯性评估：检查句子长度和结构
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # 检查是否有合理的句子长度
        avg_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        if 5 <= avg_length <= 20:
            return 1.0
        elif 3 <= avg_length <= 25:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_relevance(self, response: str, user_input: str) -> float:
        """评估回复的相关性"""
        # 检查回复是否包含用户输入中的关键词
        user_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        
        if not user_words:
            return 0.0
        
        overlap = len(user_words.intersection(response_words))
        return min(overlap / len(user_words), 1.0)
    
    def _evaluate_helpfulness(self, response: str, expected_response: str) -> float:
        """评估回复的有用性"""
        # 基于与期望回复的相似度
        return self._calculate_similarity(response, expected_response)
    
    def comprehensive_evaluation(self, test_data_path: str) -> Dict[str, Any]:
        """综合评估"""
        print("Loading test data...")
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_data.append(json.loads(line))
        
        print("Starting comprehensive evaluation...")
        
        # 领域理解评估
        print("Evaluating domain understanding...")
        domain_accuracy = self.evaluate_domain_understanding(test_data)
        
        # 槽位填充评估
        print("Evaluating slot filling...")
        slot_accuracy = self.evaluate_slot_filling(test_data)
        
        # 回复质量评估
        print("Evaluating response quality...")
        quality_scores = self.evaluate_response_quality(test_data)
        
        # 计算总体分数
        overall_domain_accuracy = np.mean(list(domain_accuracy.values()))
        overall_slot_accuracy = np.mean(list(slot_accuracy.values()))
        overall_quality = np.mean(list(quality_scores.values()))
        
        results = {
            "domain_accuracy": domain_accuracy,
            "slot_accuracy": slot_accuracy,
            "quality_scores": quality_scores,
            "overall_scores": {
                "domain_accuracy": overall_domain_accuracy,
                "slot_accuracy": overall_slot_accuracy,
                "quality_score": overall_quality,
                "combined_score": (overall_domain_accuracy + overall_slot_accuracy + overall_quality) / 3
            }
        }
        
        return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate distilled model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the distilled model")
    parser.add_argument("--test_data", type=str, default="processed_data/validation.json", help="Path to test data")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path)
    
    # 执行评估
    results = evaluator.comprehensive_evaluation(args.test_data)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print("\n=== Evaluation Results ===")
    print(f"Overall Domain Accuracy: {results['overall_scores']['domain_accuracy']:.3f}")
    print(f"Overall Slot Accuracy: {results['overall_scores']['slot_accuracy']:.3f}")
    print(f"Overall Quality Score: {results['overall_scores']['quality_score']:.3f}")
    print(f"Combined Score: {results['overall_scores']['combined_score']:.3f}")
    
    print("\nDomain-wise Results:")
    for domain, accuracy in results['domain_accuracy'].items():
        print(f"  {domain}: {accuracy:.3f}")

if __name__ == "__main__":
    main()
