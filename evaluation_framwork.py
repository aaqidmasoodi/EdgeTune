import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
import time
from collections import defaultdict
import torch

class QueryRewritingEvaluator:
    """Comprehensive evaluation framework for query rewriting"""
    
    def __init__(self, reranker_model="BAAI/bge-reranker-large"):
        """Initialize evaluator with BGE reranker"""
        print(f"Loading reranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.evaluation_results = {}
        
    def compute_relevance_scores(self, queries: List[str], documents: List[List[str]]) -> List[List[float]]:
        """Compute relevance scores using BGE reranker"""
        all_scores = []
        
        for query, doc_list in zip(queries, documents):
            if not doc_list:
                all_scores.append([])
                continue
                
            # Create query-document pairs
            pairs = [(query, doc) for doc in doc_list]
            
            # Get relevance scores
            scores = self.reranker.predict(pairs)
            all_scores.append(scores.tolist() if hasattr(scores, 'tolist') else scores)
            
        return all_scores
    
    def compute_mrr(self, relevance_scores: List[List[float]], threshold: float = 0.5) -> float:
        """Compute Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for scores in relevance_scores:
            if not scores:
                reciprocal_ranks.append(0.0)
                continue
                
            # Find first relevant document (score > threshold)
            first_relevant_rank = None
            for rank, score in enumerate(scores, 1):
                if score > threshold:
                    first_relevant_rank = rank
                    break
            
            if first_relevant_rank is not None:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def compute_recall_at_k(self, relevance_scores: List[List[float]], k: int = 10, threshold: float = 0.5) -> float:
        """Compute Recall@K"""
        recall_scores = []
        
        for scores in relevance_scores:
            if not scores:
                recall_scores.append(0.0)
                continue
            
            # Consider top-k documents
            top_k_scores = scores[:k]
            relevant_in_top_k = sum(1 for score in top_k_scores if score > threshold)
            total_relevant = sum(1 for score in scores if score > threshold)
            
            if total_relevant > 0:
                recall_scores.append(relevant_in_top_k / total_relevant)
            else:
                recall_scores.append(0.0)
        
        return np.mean(recall_scores)
    
    def compute_ndcg_at_k(self, relevance_scores: List[List[float]], k: int = 10) -> float:
        """Compute Normalized Discounted Cumulative Gain at K"""
        ndcg_scores = []
        
        for scores in relevance_scores:
            if not scores:
                ndcg_scores.append(0.0)
                continue
            
            # DCG calculation
            dcg = 0.0
            for i, score in enumerate(scores[:k]):
                dcg += score / np.log2(i + 2)  # i + 2 because log2(1) = 0
            
            # IDCG calculation (perfect ranking)
            sorted_scores = sorted(scores, reverse=True)
            idcg = 0.0
            for i, score in enumerate(sorted_scores[:k]):
                idcg += score / np.log2(i + 2)
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores)
    
    def compute_semantic_similarity(self, original_queries: List[str], rewritten_queries: List[str]) -> float:
        """Compute semantic similarity between original and rewritten queries"""
        if len(original_queries) != len(rewritten_queries):
            raise ValueError("Original and rewritten queries must have same length")
        
        # Get embeddings
        original_embeddings = self.sentence_model.encode(original_queries)
        rewritten_embeddings = self.sentence_model.encode(rewritten_queries)
        
        # Compute cosine similarities
        similarities = []
        for orig_emb, rewr_emb in zip(original_embeddings, rewritten_embeddings):
            similarity = cosine_similarity([orig_emb], [rewr_emb])[0][0]
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def compute_query_quality_metrics(self, queries: List[str]) -> Dict[str, float]:
        """Compute query quality metrics"""
        metrics = {
            'avg_length': np.mean([len(q.split()) for q in queries]),
            'avg_char_length': np.mean([len(q) for q in queries]),
            'unique_queries_ratio': len(set(queries)) / len(queries),
            'question_ratio': sum(1 for q in queries if '?' in q) / len(queries)
        }
        return metrics
    
    def evaluate_optimization_method(self, method_name: str, original_queries: List[str], 
                                   rewritten_queries: List[str], documents: List[List[str]],
                                   inference_times: List[float], memory_usage_mb: float) -> Dict:
        """Comprehensive evaluation of an optimization method"""
        
        print(f"Evaluating {method_name}...")
        
        # Compute relevance scores for original and rewritten queries
        print(f"Computing relevance scores for {len(original_queries)} queries...")
        original_scores = self.compute_relevance_scores(original_queries, documents)
        rewritten_scores = self.compute_relevance_scores(rewritten_queries, documents)
        
        # Compute retrieval metrics
        original_mrr = self.compute_mrr(original_scores)
        rewritten_mrr = self.compute_mrr(rewritten_scores)
        
        original_recall_10 = self.compute_recall_at_k(original_scores, k=10)
        rewritten_recall_10 = self.compute_recall_at_k(rewritten_scores, k=10)
        
        original_ndcg_10 = self.compute_ndcg_at_k(original_scores, k=10)
        rewritten_ndcg_10 = self.compute_ndcg_at_k(rewritten_scores, k=10)
        
        # Compute semantic similarity
        semantic_similarity = self.compute_semantic_similarity(original_queries, rewritten_queries)
        
        # Compute query quality metrics
        original_quality = self.compute_query_quality_metrics(original_queries)
        rewritten_quality = self.compute_query_quality_metrics(rewritten_queries)
        
        # Performance metrics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        
        # Compile results
        results = {
            'method': method_name,
            'retrieval_metrics': {
                'original_mrr': original_mrr,
                'rewritten_mrr': rewritten_mrr,
                'mrr_improvement': rewritten_mrr - original_mrr,
                'original_recall_10': original_recall_10,
                'rewritten_recall_10': rewritten_recall_10,
                'recall_10_improvement': rewritten_recall_10 - original_recall_10,
                'original_ndcg_10': original_ndcg_10,
                'rewritten_ndcg_10': rewritten_ndcg_10,
                'ndcg_10_improvement': rewritten_ndcg_10 - original_ndcg_10
            },
            'semantic_metrics': {
                'query_similarity': semantic_similarity,
                'original_query_quality': original_quality,
                'rewritten_query_quality': rewritten_quality
            },
            'performance_metrics': {
                'avg_inference_time': avg_inference_time,
                'std_inference_time': std_inference_time,
                'memory_usage_mb': memory_usage_mb,
                'queries_per_second': 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            }
        }
        
        # Store results
        self.evaluation_results[method_name] = results
        
        print(f"Evaluation complete for {method_name}")
        print(f"  MRR improvement: {results['retrieval_metrics']['mrr_improvement']:.4f}")
        print(f"  Recall@10 improvement: {results['retrieval_metrics']['recall_10_improvement']:.4f}")
        print(f"  Avg inference time: {avg_inference_time:.4f}s")
        print(f"  Memory usage: {memory_usage_mb:.2f}MB")
        
        return results
    
    def compare_methods(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Create comparison table of all methods"""
        
        comparison_data = []
        
        for method_name, results in results_dict.items():
            row = {
                'Method': method_name.upper(),
                'MRR (Original)': results['retrieval_metrics']['original_mrr'],
                'MRR (Rewritten)': results['retrieval_metrics']['rewritten_mrr'],
                'MRR Improvement': results['retrieval_metrics']['mrr_improvement'],
                'Recall@10 (Original)': results['retrieval_metrics']['original_recall_10'],
                'Recall@10 (Rewritten)': results['retrieval_metrics']['rewritten_recall_10'],
                'Recall@10 Improvement': results['retrieval_metrics']['recall_10_improvement'],
                'NDCG@10 (Original)': results['retrieval_metrics']['original_ndcg_10'],
                'NDCG@10 (Rewritten)': results['retrieval_metrics']['rewritten_ndcg_10'],
                'NDCG@10 Improvement': results['retrieval_metrics']['ndcg_10_improvement'],
                'Semantic Similarity': results['semantic_metrics']['query_similarity'],
                'Avg Inference Time (s)': results['performance_metrics']['avg_inference_time'],
                'Memory Usage (MB)': results['performance_metrics']['memory_usage_mb'],
                'Queries/Second': results['performance_metrics']['queries_per_second']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_performance_comparison(self, results_dict: Dict[str, Dict], save_path: str = None):
        """Create performance comparison plots"""
        
        methods = list(results_dict.keys())
        
        # Prepare data for plotting
        mrr_improvements = [results_dict[m]['retrieval_metrics']['mrr_improvement'] for m in methods]
        recall_improvements = [results_dict[m]['retrieval_metrics']['recall_10_improvement'] for m in methods]
        inference_times = [results_dict[m]['performance_metrics']['avg_inference_time'] for m in methods]
        memory_usage = [results_dict[m]['performance_metrics']['memory_usage_mb'] for m in methods]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # MRR Improvement
        bars1 = ax1.bar(methods, mrr_improvements, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('MRR Improvement by Method', fontsize=14, fontweight='bold')
        ax1.set_ylabel('MRR Improvement')
        ax1.set_ylim(min(mrr_improvements) - 0.01, max(mrr_improvements) + 0.01)
        for bar, val in zip(bars1, mrr_improvements):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom')
        
        # Recall@10 Improvement
        bars2 = ax2.bar(methods, recall_improvements, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Recall@10 Improvement by Method', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Recall@10 Improvement')
        ax2.set_ylim(min(recall_improvements) - 0.01, max(recall_improvements) + 0.01)
        for bar, val in zip(bars2, recall_improvements):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom')
        
        # Inference Time
        bars3 = ax3.bar(methods, inference_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_title('Average Inference Time by Method', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Inference Time (seconds)')
        ax3.set_ylim(0, max(inference_times) * 1.1)
        for bar, val in zip(bars3, inference_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(inference_times) * 0.01, 
                    f'{val:.4f}s', ha='center', va='bottom')
        
        # Memory Usage
        bars4 = ax4.bar(methods, memory_usage, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax4.set_title('Memory Usage by Method', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_ylim(0, max(memory_usage) * 1.1)
        for bar, val in zip(bars4, memory_usage):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_usage) * 0.01, 
                    f'{val:.1f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_trade_off_analysis(self, results_dict: Dict[str, Dict], save_path: str = None):
        """Create trade-off analysis plots"""
        
        methods = list(results_dict.keys())
        
        # Extract metrics for trade-off analysis
        mrr_improvements = [results_dict[m]['retrieval_metrics']['mrr_improvement'] for m in methods]
        inference_times = [results_dict[m]['performance_metrics']['avg_inference_time'] for m in methods]
        memory_usage = [results_dict[m]['performance_metrics']['memory_usage_mb'] for m in methods]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs Latency Trade-off
        scatter1 = ax1.scatter(inference_times, mrr_improvements, 
                              s=200, alpha=0.7, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_xlabel('Average Inference Time (seconds)')
        ax1.set_ylabel('MRR Improvement')
        ax1.set_title('Accuracy vs Latency Trade-off', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add method labels
        for i, method in enumerate(methods):
            ax1.annotate(method.upper(), (inference_times[i], mrr_improvements[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # Accuracy vs Memory Trade-off
        scatter2 = ax2.scatter(memory_usage, mrr_improvements, 
                              s=200, alpha=0.7, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_xlabel('Memory Usage (MB)')
        ax2.set_ylabel('MRR Improvement')
        ax2.set_title('Accuracy vs Memory Trade-off', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add method labels
        for i, method in enumerate(methods):
            ax2.annotate(method.upper(), (memory_usage[i], mrr_improvements[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trade-off analysis plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results_dict: Dict[str, Dict], save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("=" * 80)
        report.append("QUERY REWRITING OPTIMIZATION METHODS - EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        best_mrr_method = max(results_dict.keys(), 
                             key=lambda x: results_dict[x]['retrieval_metrics']['mrr_improvement'])
        best_speed_method = min(results_dict.keys(), 
                               key=lambda x: results_dict[x]['performance_metrics']['avg_inference_time'])
        best_memory_method = min(results_dict.keys(), 
                                key=lambda x: results_dict[x]['performance_metrics']['memory_usage_mb'])
        
        report.append(f"• Best Accuracy (MRR Improvement): {best_mrr_method.upper()}")
        report.append(f"• Fastest Method: {best_speed_method.upper()}")
        report.append(f"• Most Memory Efficient: {best_memory_method.upper()}")
        report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS BY METHOD")
        report.append("-" * 40)
        
        for method_name, results in results_dict.items():
            report.append(f"\n{method_name.upper()} METHOD:")
            report.append("  Retrieval Performance:")
            report.append(f"    • MRR Improvement: {results['retrieval_metrics']['mrr_improvement']:.4f}")
            report.append(f"    • Recall@10 Improvement: {results['retrieval_metrics']['recall_10_improvement']:.4f}")
            report.append(f"    • NDCG@10 Improvement: {results['retrieval_metrics']['ndcg_10_improvement']:.4f}")
            report.append("  Computational Performance:")
            report.append(f"    • Average Inference Time: {results['performance_metrics']['avg_inference_time']:.4f}s")
            report.append(f"    • Memory Usage: {results['performance_metrics']['memory_usage_mb']:.2f}MB")
            report.append(f"    • Throughput: {results['performance_metrics']['queries_per_second']:.2f} queries/sec")
            report.append("  Semantic Quality:")
            report.append(f"    • Query Similarity: {results['semantic_metrics']['query_similarity']:.4f}")
        
        # Performance Comparison Table
        report.append("\n\nPERFORMANCE COMPARISON TABLE")
        report.append("-" * 40)
        comparison_df = self.compare_methods(results_dict)
        report.append(comparison_df.to_string(index=False))
        
        # Recommendations
        report.append("\n\nRECOMMENDations")
        report.append("-" * 40)
        
        # Find balanced method (considering accuracy and efficiency)
        efficiency_scores = {}
        for method, results in results_dict.items():
            accuracy_score = results['retrieval_metrics']['mrr_improvement']
            efficiency_score = 1.0 / (results['performance_metrics']['avg_inference_time'] + 0.001)
            memory_score = 1.0 / (results['performance_metrics']['memory_usage_mb'] + 1.0)
            combined_score = accuracy_score + 0.3 * efficiency_score + 0.2 * memory_score
            efficiency_scores[method] = combined_score
        
        best_balanced_method = max(efficiency_scores.keys(), key=lambda x: efficiency_scores[x])
        
        report.append(f"• For balanced performance: {best_balanced_method.upper()}")
        report.append(f"• For maximum accuracy: {best_mrr_method.upper()}")
        report.append(f"• For real-time applications: {best_speed_method.upper()}")
        report.append(f"• For resource-constrained devices: {best_memory_method.upper()}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")
        
        return report_text
    
    def save_results_json(self, results_dict: Dict[str, Dict], save_path: str):
        """Save evaluation results to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {save_path}")

class StatisticalAnalyzer:
    """Statistical analysis for optimization method comparison"""
    
    @staticmethod
    def perform_significance_test(results_dict: Dict[str, Dict], metric: str = 'mrr_improvement'):
        """Perform statistical significance test between methods"""
        from scipy import stats
        
        # Extract metric values (assuming we have multiple runs)
        method_values = {}
        for method, results in results_dict.items():
            if metric in results['retrieval_metrics']:
                # For simulation, create multiple samples around the mean
                base_value = results['retrieval_metrics'][metric]
                # Simulate variation
                values = np.random.normal(base_value, abs(base_value) * 0.1, 30)
                method_values[method] = values
        
        # Perform pairwise t-tests
        methods = list(method_values.keys())
        p_values = {}
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                t_stat, p_value = stats.ttest_ind(method_values[method1], method_values[method2])
                p_values[f"{method1}_vs_{method2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return p_values

# TODO testing.. ... 
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = QueryRewritingEvaluator()
    
    # Sample data for testing
    sample_queries = [
        "machine learning algorithms",
        "climate change effects",
        "quantum computing applications",
        "artificial intelligence ethics",
        "renewable energy sources"
    ]
    
    sample_rewrites = [
        "advanced machine learning algorithms and techniques",
        "environmental impacts of climate change",
        "practical applications of quantum computing",
        "ethical considerations in artificial intelligence",
        "sustainable renewable energy technologies"
    ]
    
    sample_documents = [
        ["Machine learning is a subset of AI...", "Algorithms are step-by-step procedures..."],
        ["Climate change refers to long-term shifts...", "Environmental effects include rising temperatures..."],
        ["Quantum computing uses quantum mechanics...", "Applications include cryptography and optimization..."],
        ["AI ethics deals with moral implications...", "Ethical frameworks guide AI development..."],
        ["Renewable energy comes from natural sources...", "Solar and wind are common renewable sources..."]
    ]
    
    sample_inference_times = [0.1, 0.15, 0.08, 0.12, 0.11]
    
    # Evaluate a sample method
    results = evaluator.evaluate_optimization_method(
        method_name="lora",
        original_queries=sample_queries,
        rewritten_queries=sample_rewrites,
        documents=sample_documents,
        inference_times=sample_inference_times,
        memory_usage_mb=150.5
    )
    
    print("\nSample evaluation completed!")
    print(f"MRR improvement: {results['retrieval_metrics']['mrr_improvement']:.4f}")
    print(f"Semantic similarity: {results['semantic_metrics']['query_similarity']:.4f}")