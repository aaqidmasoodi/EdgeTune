import torch
import torch.nn as nn
from transformers import (
    RobertaForSequenceClassification, 
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from copy import deepcopy
import time
import psutil
import os

class BaseOptimizer:
    """Base class for derivative-free optimization methods"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimization_history = []
        
    def measure_memory_usage(self):
        """Measure current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def measure_inference_time(self, query: str, n_runs: int = 10):
        """Measure average inference time"""
        times = []
        for _ in range(n_runs):
            start_time = time.time()
            self.generate_rewrite(query)
            end_time = time.time()
            times.append(end_time - start_time)
        return np.mean(times)
    
    def generate_rewrite(self, query: str):
        """Generate query rewrite - to be implemented by subclasses"""
        raise NotImplementedError

class LoRAOptimizer(BaseOptimizer):
    """Low-Rank Adaptation optimization"""
    
    def __init__(self, model, tokenizer, device='cpu', r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__(model, tokenizer, device)
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.setup_lora()
        
    def setup_lora(self):
        """Apply LoRA configuration to model"""
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q", "v", "k", "o"]
        )
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA applied with r={self.r}, alpha={self.lora_alpha}")
        
    def generate_rewrite(self, query: str):
        """Generate query rewrite using LoRA-adapted model"""
        inputs = self.tokenizer(
            f"Rewrite this query: {query}", 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=2,
                temperature=0.7,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class BitFitOptimizer(BaseOptimizer):
    """BitFit: Simple Parameter-Efficient Fine-Tuning"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        super().__init__(model, tokenizer, device)
        self.setup_bitfit()
        
    def setup_bitfit(self):
        """Freeze all parameters except bias terms"""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if 'bias' in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                
        print(f"BitFit: {trainable_params}/{total_params} parameters trainable "
              f"({100 * trainable_params / total_params:.2f}%)")
        
    def generate_rewrite(self, query: str):
        """Generate query rewrite using BitFit model"""
        inputs = self.tokenizer(
            f"Rewrite this query: {query}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=2,
                temperature=0.7,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class ZerothOrderOptimizer(BaseOptimizer):
    """Zeroth-Order Optimization (ZOO) implementation"""
    
    def __init__(self, model, tokenizer, device='cpu', mu=1e-3, sigma=1e-3, learning_rate=1e-4):
        super().__init__(model, tokenizer, device)
        self.mu = mu  # Smoothing parameter
        self.sigma = sigma  # Perturbation scale
        self.learning_rate = learning_rate
        self.best_params = None
        self.best_loss = float('inf')
        
    def estimate_gradient(self, loss_fn, params, data_batch):
        """Estimate gradient using finite differences"""
        gradients = {}
        
        for name, param in params.items():
            if param.requires_grad:
                # Forward perturbation
                param.data += self.sigma
                loss_plus = loss_fn(data_batch)
                
                # Backward perturbation
                param.data -= 2 * self.sigma
                loss_minus = loss_fn(data_batch)
                
                # Restore original parameter
                param.data += self.sigma
                
                # Estimate gradient
                grad_estimate = (loss_plus - loss_minus) / (2 * self.sigma)
                gradients[name] = grad_estimate
                
        return gradients
    
    def update_parameters(self, gradients):
        """Update parameters using estimated gradients"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in gradients and param.requires_grad:
                    param.data -= self.learning_rate * gradients[name]
    
    def generate_rewrite(self, query: str):
        """Generate query rewrite using ZOO-optimized model"""
        inputs = self.tokenizer(
            f"Rewrite this query: {query}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=2,
                temperature=0.7,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class EvolutionStrategyOptimizer(BaseOptimizer):
    """Evolution Strategy (ES) optimization"""
    
    def __init__(self, model, tokenizer, device='cpu', population_size=10, 
                 mutation_strength=0.1, learning_rate=0.01):
        super().__init__(model, tokenizer, device)
        self.population_size = population_size
        self.mutation_strength = mutation_strength
        self.learning_rate = learning_rate
        self.population = []
        self.fitness_scores = []
        
    def initialize_population(self):
        """Initialize population of model parameters"""
        base_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        self.population = []
        for _ in range(self.population_size):
            individual = {}
            for name, param in base_params.items():
                if param.requires_grad:
                    noise = torch.randn_like(param) * self.mutation_strength
                    individual[name] = param + noise
                else:
                    individual[name] = param
            self.population.append(individual)
    
    def evaluate_fitness(self, individual, data_batch):
        """Evaluate fitness of an individual"""
        # Temporarily set model parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.clone()
            if name in individual:
                param.data = individual[name]
        
        # Evaluate on data batch
        fitness = self.compute_loss(data_batch)
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
            
        return fitness
    
    def compute_loss(self, data_batch):
        """Compute loss for current model state"""
        # Simplified loss computation
        total_loss = 0
        for item in data_batch:
            query = item['original_query']
            target = item['rewritten_query']
            
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True)
            targets = self.tokenizer(target, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=targets.input_ids)
                total_loss += outputs.loss.item()
                
        return total_loss / len(data_batch)
    
    def update_population(self):
        """Update population based on fitness scores"""
        # Select best individuals
        sorted_indices = np.argsort(self.fitness_scores)
        elite_size = self.population_size // 2
        
        new_population = []
        # Keep elite
        for i in range(elite_size):
            new_population.append(self.population[sorted_indices[i]])
        
        # Generate offspring
        for _ in range(self.population_size - elite_size):
            parent1 = self.population[sorted_indices[np.random.randint(elite_size)]]
            parent2 = self.population[sorted_indices[np.random.randint(elite_size)]]
            
            offspring = {}
            for name in parent1:
                if np.random.random() < 0.5:
                    offspring[name] = parent1[name] + torch.randn_like(parent1[name]) * self.mutation_strength
                else:
                    offspring[name] = parent2[name] + torch.randn_like(parent2[name]) * self.mutation_strength
            
            new_population.append(offspring)
        
        self.population = new_population
    
    def generate_rewrite(self, query: str):
        """Generate query rewrite using ES-optimized model"""
        inputs = self.tokenizer(
            f"Rewrite this query: {query}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=2,
                temperature=0.7,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class OptimizationComparator:
    """Compare different optimization methods"""
    
    def __init__(self, base_model_name="t5-small"):
        self.base_model_name = base_model_name
        self.optimizers = {}
        self.results = {}
        
    def initialize_optimizers(self):
        """Initialize all optimization methods"""
        # Load base model for each optimizer
        models = {}
        tokenizers = {}
        
        for method in ['lora', 'bitfit', 'zoo', 'es']:
            if method == 'lora':
                model = T5ForConditionalGeneration.from_pretrained(self.base_model_name)
                tokenizer = T5Tokenizer.from_pretrained(self.base_model_name)
                models[method] = model
                tokenizers[method] = tokenizer
                self.optimizers[method] = LoRAOptimizer(model, tokenizer)
            elif method == 'bitfit':
                model = T5ForConditionalGeneration.from_pretrained(self.base_model_name)
                tokenizer = T5Tokenizer.from_pretrained(self.base_model_name)
                models[method] = model
                tokenizers[method] = tokenizer
                self.optimizers[method] = BitFitOptimizer(model, tokenizer)
            elif method == 'zoo':
                model = T5ForConditionalGeneration.from_pretrained(self.base_model_name)
                tokenizer = T5Tokenizer.from_pretrained(self.base_model_name)
                models[method] = model
                tokenizers[method] = tokenizer
                self.optimizers[method] = ZerothOrderOptimizer(model, tokenizer)
            elif method == 'es':
                model = T5ForConditionalGeneration.from_pretrained(self.base_model_name)
                tokenizer = T5Tokenizer.from_pretrained(self.base_model_name)
                models[method] = model
                tokenizers[method] = tokenizer
                self.optimizers[method] = EvolutionStrategyOptimizer(model, tokenizer)
        
        print("All optimizers initialized successfully!")
    
    def run_comparison(self, test_queries: List[str]):
        """Run comparison across all methods"""
        results = {}
        
        for method_name, optimizer in self.optimizers.items():
            print(f"\nEvaluating {method_name.upper()}...")
            
            # Measure performance metrics
            start_memory = optimizer.measure_memory_usage()
            
            rewrites = []
            inference_times = []
            
            for query in test_queries[:10]:  # Test on subset
                start_time = time.time()
                rewrite = optimizer.generate_rewrite(query)
                end_time = time.time()
                
                rewrites.append(rewrite)
                inference_times.append(end_time - start_time)
            
            end_memory = optimizer.measure_memory_usage()
            
            results[method_name] = {
                'rewrites': rewrites,
                'avg_inference_time': np.mean(inference_times),
                'memory_usage_mb': end_memory - start_memory,
                'std_inference_time': np.std(inference_times)
            }
            
            print(f"{method_name} - Avg inference time: {results[method_name]['avg_inference_time']:.4f}s")
            print(f"{method_name} - Memory usage: {results[method_name]['memory_usage_mb']:.2f}MB")
        
        self.results = results
        return results

if __name__ == "__main__":
    # Example usage
    comparator = OptimizationComparator()
    comparator.initialize_optimizers()
    
    test_queries = [
        "machine learning algorithms",
        "climate change effects",
        "quantum computing applications",
        "artificial intelligence ethics",
        "renewable energy sources"
    ]
    
    results = comparator.run_comparison(test_queries)
    
    print("\n=== COMPARISON RESULTS ===")
    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Average inference time: {metrics['avg_inference_time']:.4f}s")
        print(f"  Memory usage: {metrics['memory_usage_mb']:.2f}MB")
        print(f"  Example rewrite: {metrics['rewrites'][0]}")