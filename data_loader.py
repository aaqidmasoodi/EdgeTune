import os
import json
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Tuple
import random

class MSMARCOQueryRewritingDataset(Dataset):
    """Dataset class for MS MARCO query rewriting task"""
    
    def __init__(self, queries: List[str], passages: List[List[str]], 
                 rewritten_queries: List[str] = None, max_length: int = 512):
        self.queries = queries
        self.passages = passages
        self.rewritten_queries = rewritten_queries or queries 
        self.max_length = max_length
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return {
            'original_query': self.queries[idx],
            'passages': self.passages[idx],
            'rewritten_query': self.rewritten_queries[idx]
        }

def custom_collate_fn(batch):
    """Custom collate function to handle lists of variable-length passages"""
    original_queries = [item['original_query'] for item in batch]
    rewritten_queries = [item['rewritten_query'] for item in batch]
    passages = [item['passages'] for item in batch]  # List[List[str]]
    
    return {
        'original_query': original_queries,
        'rewritten_query': rewritten_queries,
        'passages': passages
    }

class MSMARCODataLoader:
    """Data loader and preprocessor for MS MARCO dataset"""
    
    def __init__(self, subset_size: int = 60000, random_seed: int = 42):
        self.subset_size = subset_size
        self.random_seed = random_seed
        random.seed(random_seed)
        
    def load_ms_marco_data(self) -> Tuple[List[str], List[List[str]], List[int]]:
        """Load and preprocess MS MARCO dataset"""
        
        print("Loading MS MARCO dataset...")
        
        # Load the dataset
        dataset = load_dataset("ms_marco", "v1.1", split="train")
        
        # Sample subset
        if self.subset_size < len(dataset):
            indices = random.sample(range(len(dataset)), self.subset_size)
            dataset = dataset.select(indices)
        
        queries = []
        passages = []
        relevance_labels = []
        
        print(f"Processing {len(dataset)} samples...")
        
        for item in dataset:
            query = item['query']
            passage_list = item['passages']['passage_text']
            is_selected = item['passages']['is_selected']
            
            # Filter relevant passages
            relevant_passages = [p for p, selected in zip(passage_list, is_selected) if selected == 1]
            
            if relevant_passages:  # Only include queries with relevant passages
                queries.append(query)
                passages.append(relevant_passages)
                relevance_labels.append(1)  # Has relevant passages
        
        print(f"Loaded {len(queries)} queries with relevant passages")
        return queries, passages, relevance_labels
    
    def create_synthetic_rewrites(self, queries: List[str]) -> List[str]:
        """Create synthetic query rewrites for training data"""
        
        rewrite_templates = [
            lambda q: f"Find information about {q}",
            lambda q: f"Search for {q}",
            lambda q: f"What is {q}?",
            lambda q: f"Tell me about {q}",
            lambda q: f"Explain {q}",
            lambda q: f"{q} information",
            lambda q: f"Details on {q}",
            lambda q: f"Learn about {q}"
        ]
        
        rewrites = []
        for query in queries:
            if random.random() < 0.7:  # 70% chance to rewrite
                template = random.choice(rewrite_templates)
                rewrite = template(query)
            else:
                rewrite = query
            rewrites.append(rewrite)
            
        return rewrites
    
    def split_data(self, queries: List[str], passages: List[List[str]], 
                   train_ratio: float = 0.8) -> Dict:
        """Split data into train/validation sets"""
        
        n_samples = len(queries)
        n_train = int(n_samples * train_ratio)
        
        # Shuffle indices
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create synthetic rewrites for training
        rewritten_queries = self.create_synthetic_rewrites(queries)
        
        return {
            'train': {
                'queries': [queries[i] for i in train_indices],
                'passages': [passages[i] for i in train_indices],
                'rewrites': [rewritten_queries[i] for i in train_indices]
            },
            'val': {
                'queries': [queries[i] for i in val_indices],
                'passages': [passages[i] for i in val_indices],
                'rewrites': [rewritten_queries[i] for i in val_indices]
            }
        }
    
    def create_dataloaders(self, batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders"""
        
        # Load and split data
        queries, passages, _ = self.load_ms_marco_data()
        data_split = self.split_data(queries, passages)
        
        # Create datasets
        train_dataset = MSMARCOQueryRewritingDataset(
            data_split['train']['queries'],
            data_split['train']['passages'],
            data_split['train']['rewrites']
        )
        
        val_dataset = MSMARCOQueryRewritingDataset(
            data_split['val']['queries'],
            data_split['val']['passages'],
            data_split['val']['rewrites']
        )
        
        # Create dataloaders with custom collate_fn
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        return train_loader, val_loader


if __name__ == "__main__":
    loader = MSMARCODataLoader(subset_size=1000)  # Small subset for testing
    train_loader, val_loader = loader.create_dataloaders(batch_size=8)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test loading a batch
    for batch in train_loader:
        print("Sample batch:")
        print(f"Original query: {batch['original_query'][0]}")
        print(f"Rewritten query: {batch['rewritten_query'][0]}")
        print(f"Number of passages: {len(batch['passages'][0])}")
        break
