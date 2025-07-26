import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# custom modules
from data_loader import MSMARCODataLoader
from optimisation_methods import OptimizationComparator
from evaluation_framwork import QueryRewritingEvaluator, StatisticalAnalyzer
# from on_device_sim import simulate_on_device_environment # I have removed this.. (It only turns off CUDA)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Main experiment runner for the practicum project"""
    
    def __init__(self, config_path: str = None):
        """Initialize experiment runner with configuration"""
        
        self.config = self.load_config(config_path)
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.optimization_comparator = None
        self.evaluator = None
        self.results = {}
        
        logger.info("Experiment runner initialized")
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def load_config(self, config_path: str = None) -> dict:
        """Load experiment configuration"""
        
        default_config = {
            'data': {
                'subset_size': 5000,  # Reduced for faster experimentation
                'train_ratio': 0.8,
                'batch_size': 16,
                'random_seed': 42
            },
            'models': {
                'base_model': 't5-small',  # Using smaller model for device constraints
                'reranker_model': 'BAAI/bge-reranker-large'
            },
            'optimization': {
                'methods': ['lora', 'bitfit', 'zoo', 'es'],
                'lora_config': {
                    'r': 8,
                    'lora_alpha': 32,
                    'lora_dropout': 0.1
                },
                'zoo_config': {
                    'mu': 1e-3,
                    'sigma': 1e-3,
                    'learning_rate': 1e-4
                },
                'es_config': {
                    'population_size': 10,
                    'mutation_strength': 0.1,
                    'learning_rate': 0.01
                }
            },
            'evaluation': {
                'metrics': ['mrr', 'recall@10', 'ndcg@10'],
                'relevance_threshold': 0.5,
                'test_queries_limit': 100  # Limit for faster evaluation
            },
            'results_dir': 'results',
            'save_plots': True,
            'save_detailed_results': True
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Update default config with user config
            default_config.update(user_config)
        
        return default_config
    
    def setup_environment(self):
        """Setup experimental environment"""
        logger.info("Setting up experimental environment...")
        
        # Simulate on-device constraints
        #simulate_on_device_environment() # DONE MANURALLY NOW
        
        # Set random seeds for reproducibility
        seed = self.config['data']['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        logger.info("Environment setup complete")
    
    def load_data(self):
        """Load and prepare MS MARCO dataset"""
        logger.info("Loading MS MARCO dataset...")
        
        self.data_loader = MSMARCODataLoader(
            subset_size=self.config['data']['subset_size'],
            random_seed=self.config['data']['random_seed']
        )
        
        # Load data
        queries, passages, _ = self.data_loader.load_ms_marco_data()
        self.data_split = self.data_loader.split_data(queries, passages)
        
        logger.info(f"Loaded {len(queries)} queries with relevant passages")
        logger.info(f"Train set: {len(self.data_split['train']['queries'])} queries")
        logger.info(f"Validation set: {len(self.data_split['val']['queries'])} queries")
    
    def initialize_optimization_methods(self):
        """Initialize all optimization methods"""
        logger.info("Initializing optimization methods...")
        
        self.optimization_comparator = OptimizationComparator(
            base_model_name=self.config['models']['base_model']
        )
        self.optimization_comparator.initialize_optimizers()
        
        logger.info(f"Initialized {len(self.optimization_comparator.optimizers)} optimization methods")
    
    def run_optimization_experiments(self):
        """Run experiments with all optimization methods"""
        logger.info("Running optimization experiments...")
        
        # Use validation set for testing
        test_queries = self.data_split['val']['queries'][:self.config['evaluation']['test_queries_limit']]
        test_passages = self.data_split['val']['passages'][:self.config['evaluation']['test_queries_limit']]
        
        method_results = {}
        
        for method_name, optimizer in self.optimization_comparator.optimizers.items():
            logger.info(f"Testing {method_name.upper()} method...")
            
            start_time = time.time()
            
            # Generate rewrites
            rewritten_queries = []
            inference_times = []
            
            for query in test_queries:
                query_start = time.time()
                try:
                    rewrite = optimizer.generate_rewrite(query)
                    rewritten_queries.append(rewrite)
                except Exception as e:
                    logger.warning(f"Error generating rewrite for query '{query}': {e}")
                    rewritten_queries.append(query)  # Fallback to original
                
                query_end = time.time()
                inference_times.append(query_end - query_start)
            
            # Measure memory usage
            memory_usage = optimizer.measure_memory_usage()
            
            method_results[method_name] = {
                'original_queries': test_queries,
                'rewritten_queries': rewritten_queries,
                'documents': test_passages,
                'inference_times': inference_times,
                'memory_usage_mb': memory_usage,
                'total_time': time.time() - start_time
            }
            
            logger.info(f"{method_name.upper()} completed in {method_results[method_name]['total_time']:.2f}s")
            logger.info(f"Average inference time: {np.mean(inference_times):.4f}s")
            logger.info(f"Memory usage: {memory_usage:.2f}MB")
        
        self.method_results = method_results
    
    def evaluate_methods(self):
        """Evaluate all optimization methods"""
        logger.info("Evaluating optimization methods...")
        
        # Initialize evaluator
        self.evaluator = QueryRewritingEvaluator(
            reranker_model=self.config['models']['reranker_model']
        )
        
        evaluation_results = {}
        
        for method_name, method_data in self.method_results.items():
            logger.info(f"Evaluating {method_name.upper()}...")
            
            results = self.evaluator.evaluate_optimization_method(
                method_name=method_name,
                original_queries=method_data['original_queries'],
                rewritten_queries=method_data['rewritten_queries'],
                documents=method_data['documents'],
                inference_times=method_data['inference_times'],
                memory_usage_mb=method_data['memory_usage_mb']
            )
            
            evaluation_results[method_name] = results
        
        self.evaluation_results = evaluation_results
        logger.info("Evaluation completed for all methods")
    
    def generate_reports_and_plots(self):
        """Generate comprehensive reports and visualizations"""
        logger.info("Generating reports and visualizations...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate comparison table
        comparison_df = self.evaluator.compare_methods(self.evaluation_results)
        comparison_file = self.results_dir / f"method_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Comparison table saved to {comparison_file}")
        
        # Generate plots if enabled
        if self.config['save_plots']:
            performance_plot_path = self.results_dir / f"performance_comparison_{timestamp}.png"
            self.evaluator.plot_performance_comparison(
                self.evaluation_results, 
                save_path=str(performance_plot_path)
            )
            
            tradeoff_plot_path = self.results_dir / f"tradeoff_analysis_{timestamp}.png"
            self.evaluator.plot_trade_off_analysis(
                self.evaluation_results,
                save_path=str(tradeoff_plot_path)
            )
        
        # Generate comprehensive report
        report_file = self.results_dir / f"evaluation_report_{timestamp}.txt"
        report_text = self.evaluator.generate_evaluation_report(
            self.evaluation_results,
            save_path=str(report_file)
        )
        
        # Save detailed results
        if self.config['save_detailed_results']:
            results_file = self.results_dir / f"detailed_results_{timestamp}.json"
            self.evaluator.save_results_json(self.evaluation_results, str(results_file))
        
        logger.info("Reports and visualizations generated successfully")
        
        return {
            'comparison_table': comparison_df,
            'report_text': report_text,
            'evaluation_results': self.evaluation_results
        }
    
    def perform_statistical_analysis(self):
        """Perform statistical analysis on results"""
        logger.info("Performing statistical analysis...")
        
        analyzer = StatisticalAnalyzer()
        
        # Perform significance tests
        significance_results = analyzer.perform_significance_test(
            self.evaluation_results, 
            metric='mrr_improvement'
        )
        
        # Save statistical analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = self.results_dir / f"statistical_analysis_{timestamp}.json"
        
        with open(stats_file, 'w') as f:
            json.dump(significance_results, f, indent=2)
        
        logger.info(f"Statistical analysis saved to {stats_file}")
        
        # Print significance results
        logger.info("Statistical Significance Results:")
        for comparison, results in significance_results.items():
            significance = "SIGNIFICANT" if results['significant'] else "NOT SIGNIFICANT"
            logger.info(f"  {comparison}: p-value = {results['p_value']:.4f} ({significance})")
        
        return significance_results
    
    def run_complete_experiment(self):
        """Run the complete experimental pipeline"""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE EXPERIMENTAL PIPELINE")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # NOTE PLEASE DO NOT !! REMOVE THE TRY BLOCK

        try:
            # Step 1: Environment setup
            self.setup_environment()
            
            # Step 2: Data loading
            self.load_data()
            
            # Step 3: Initialize optimization methods
            self.initialize_optimization_methods()
            
            # Step 4: Run optimization experiments
            self.run_optimization_experiments()
            
            # Step 5: Evaluate methods
            self.evaluate_methods()
            
            # Step 6: Generate reports and plots
            final_results = self.generate_reports_and_plots()
            
            # Step 7: Statistical analysis
            statistical_results = self.perform_statistical_analysis()
            
            total_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info("EXPERIMENTAL PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            logger.info("=" * 80)
            
            return {
                'success': True,
                'execution_time': total_time,
                'final_results': final_results,
                'statistical_results': statistical_results
            }
            
        except Exception as e:
            logger.error(f"Experiment failed with error: {str(e)}")
            logger.error("Check the log file for detailed error information")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def print_summary(self):
        """Print experiment summary"""
        if not hasattr(self, 'evaluation_results'):
            logger.warning("No evaluation results available. Run experiment first.")
            return
        
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        
        # Find best performing methods
        methods = list(self.evaluation_results.keys())
        
        best_accuracy = max(methods, 
                           key=lambda x: self.evaluation_results[x]['retrieval_metrics']['mrr_improvement'])
        best_speed = min(methods, 
                        key=lambda x: self.evaluation_results[x]['performance_metrics']['avg_inference_time'])
        best_memory = min(methods, 
                         key=lambda x: self.evaluation_results[x]['performance_metrics']['memory_usage_mb'])
        
        print(f"Dataset: MS MARCO ({self.config['data']['subset_size']} samples)")
        print(f"Base Model: {self.config['models']['base_model']}")
        print(f"Test Queries: {len(self.method_results[methods[0]]['original_queries'])}")
        print()
        
        print("BEST PERFORMING METHODS:")
        print(f"  Highest Accuracy (MRR): {best_accuracy.upper()}")
        mrr_val = self.evaluation_results[best_accuracy]['retrieval_metrics']['mrr_improvement']
        print(f"    MRR Improvement: {mrr_val:.4f}")
        
        print(f"  Fastest Inference: {best_speed.upper()}")
        speed_val = self.evaluation_results[best_speed]['performance_metrics']['avg_inference_time']
        print(f"    Avg Inference Time: {speed_val:.4f}s")
        
        print(f"  Most Memory Efficient: {best_memory.upper()}")
        memory_val = self.evaluation_results[best_memory]['performance_metrics']['memory_usage_mb']
        print(f"    Memory Usage: {memory_val:.2f}MB")
        
        print()
        print("DETAILED RESULTS:")
        print("-" * 40)
        
        for method in methods:
            results = self.evaluation_results[method]
            print(f"{method.upper()}:")
            print(f"  MRR Improvement: {results['retrieval_metrics']['mrr_improvement']:.4f}")
            print(f"  Recall@10 Improvement: {results['retrieval_metrics']['recall_10_improvement']:.4f}")
            print(f"  Avg Inference Time: {results['performance_metrics']['avg_inference_time']:.4f}s")
            print(f"  Memory Usage: {results['performance_metrics']['memory_usage_mb']:.2f}MB")
            print(f"  Semantic Similarity: {results['semantic_metrics']['query_similarity']:.4f}")
            print()

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "data": {
            "subset_size": 5000,
            "train_ratio": 0.8,
            "batch_size": 16,
            "random_seed": 42
        },
        "models": {
            "base_model": "t5-small",
            "reranker_model": "BAAI/bge-reranker-large"
        },
        "optimization": {
            "methods": ["lora", "bitfit", "zoo", "es"],
            "lora_config": {
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1
            }
        },
        "evaluation": {
            "test_queries_limit": 100,
            "relevance_threshold": 0.5
        },
        "results_dir": "results",
        "save_plots": True,
        "save_detailed_results": True
    }
    
    with open('experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration saved to 'experiment_config.json'")

def main():

    import argparse
    
    parser = argparse.ArgumentParser(description='Query Rewriting Optimization Experiment')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create sample configuration file')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced dataset')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    # Initialize experiment runner
    runner = ExperimentRunner(config_path=args.config)
    
    # Modify config for quick test
    if args.quick_test:
        runner.config['data']['subset_size'] = 1000
        runner.config['evaluation']['test_queries_limit'] = 20
        logger.info("Running in quick test mode with reduced dataset")
    
    # Run complete experiment
    results = runner.run_complete_experiment()
    
    if results['success']:
        runner.print_summary()
        print(f"\nAll results saved to: {runner.results_dir}")
        print("Experiment completed successfully!")
    else:
        print(f"Experiment failed: {results['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())