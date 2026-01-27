import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils
from datetime import datetime


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'compare_models_{timestamp}.log')
logger = utils.setup_logger('compare_models', log_file=log_file)


def load_test_results():
    single_results_path = 'test_results_summary.yaml'
    composite_results_path = 'test_composite_results_summary.yaml'
    
    single_results = None
    composite_results = None
    
    if os.path.exists(single_results_path):
        single_results = utils.load_yaml(single_results_path)
        logger.info(f"Loaded single-model results from {single_results_path}")
    else:
        logger.warning(f"Single-model results not found: {single_results_path}")
    
    if os.path.exists(composite_results_path):
        composite_results = utils.load_yaml(composite_results_path)
        logger.info(f"Loaded composite-model results from {composite_results_path}")
    else:
        logger.warning(f"Composite-model results not found: {composite_results_path}")
    
    return single_results, composite_results


def calculate_overall_metrics(results):
    if not results:
        return None
    
    all_metrics = []
    for particle_or_dataset, dataset_results in results.items():
        if isinstance(dataset_results, dict):
            for dataset_type, result in dataset_results.items():
                if result and 'metrics' in result:
                    all_metrics.append(result['metrics'])
    
    if not all_metrics:
        return None
    
    return {
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'total_tp': sum([m['total_tp'] for m in all_metrics]),
        'total_fp': sum([m['total_fp'] for m in all_metrics]),
        'total_fn': sum([m['total_fn'] for m in all_metrics])
    }


def print_comparison(single_results, composite_results):
    logger.info("\n" + "="*80)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info("="*80)
    
    if single_results:
        logger.info("\n--- SINGLE-MODEL APPROACH ---")
        for particle_type, dataset_results in single_results.items():
            logger.info(f"\n{particle_type} Model:")
            
            particle_metrics = []
            for dataset_type, result in dataset_results.items():
                if result and 'metrics' in result:
                    metrics = result['metrics']
                    particle_metrics.append(metrics)
                    logger.info(f"  {dataset_type}:")
                    logger.info(f"    Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
            
            if particle_metrics:
                avg_p = np.mean([m['precision'] for m in particle_metrics])
                avg_r = np.mean([m['recall'] for m in particle_metrics])
                avg_f1 = np.mean([m['f1_score'] for m in particle_metrics])
                logger.info(f"  Average: Precision={avg_p:.3f}, Recall={avg_r:.3f}, F1={avg_f1:.3f}")
    
    if composite_results:
        logger.info("\n--- COMPOSITE-MODEL APPROACH ---")
        for dataset_type, result in composite_results.items():
            if result and 'metrics' in result:
                metrics = result['metrics']
                logger.info(f"\n{dataset_type}:")
                logger.info(f"  Precision: {metrics['precision']:.3f}")
                logger.info(f"  Recall: {metrics['recall']:.3f}")
                logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
                logger.info(f"  Total TP: {metrics['total_tp']}")
                logger.info(f"  Total FP: {metrics['total_fp']}")
                logger.info(f"  Total FN: {metrics['total_fn']}")
    
    single_overall = calculate_overall_metrics(single_results)
    composite_overall = calculate_overall_metrics(composite_results)
    
    logger.info("\n" + "="*80)
    logger.info("OVERALL COMPARISON")
    logger.info("="*80)
    
    if single_overall and composite_overall:
        logger.info("\nSingle-Model Approach (Average across all particle types and datasets):")
        logger.info(f"  Precision: {single_overall['precision']:.3f}")
        logger.info(f"  Recall: {single_overall['recall']:.3f}")
        logger.info(f"  F1-Score: {single_overall['f1_score']:.3f}")
        logger.info(f"  Total TP: {single_overall['total_tp']}")
        logger.info(f"  Total FP: {single_overall['total_fp']}")
        logger.info(f"  Total FN: {single_overall['total_fn']}")
        
        logger.info("\nComposite-Model Approach (All particle types simultaneously):")
        logger.info(f"  Precision: {composite_overall['precision']:.3f}")
        logger.info(f"  Recall: {composite_overall['recall']:.3f}")
        logger.info(f"  F1-Score: {composite_overall['f1_score']:.3f}")
        logger.info(f"  Total TP: {composite_overall['total_tp']}")
        logger.info(f"  Total FP: {composite_overall['total_fp']}")
        logger.info(f"  Total FN: {composite_overall['total_fn']}")
        
        logger.info("\nDifference (Composite - Single):")
        logger.info(f"  Precision: {composite_overall['precision'] - single_overall['precision']:+.3f}")
        logger.info(f"  Recall: {composite_overall['recall'] - single_overall['recall']:+.3f}")
        logger.info(f"  F1-Score: {composite_overall['f1_score'] - single_overall['f1_score']:+.3f}")
        
        return single_overall, composite_overall
    
    return None, None


def plot_comparison(single_overall, composite_overall):
    if not single_overall or not composite_overall:
        logger.warning("Cannot create comparison plot: missing results")
        return
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    single_values = [single_overall['precision'], single_overall['recall'], single_overall['f1_score']]
    composite_values = [composite_overall['precision'], composite_overall['recall'], composite_overall['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, single_values, width, label='Single-Model', color='steelblue')
    bars2 = ax.bar(x + width/2, composite_values, width, label='Composite-Model', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_dir = 'detection_results/comparison'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\nComparison plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare single-model vs composite-model performance')
    args = parser.parse_args()
    
    logger.info("=== Model Comparison Analysis ===")
    
    single_results, composite_results = load_test_results()
    
    if not single_results and not composite_results:
        logger.error("No results found. Please run testing first.")
        logger.error("  For single models: python src/test_single_particle.py")
        logger.error("  For composite model: python src/test_composite_model.py")
        return
    
    single_overall, composite_overall = print_comparison(single_results, composite_results)
    
    if single_overall and composite_overall:
        plot_comparison(single_overall, composite_overall)
    
    logger.info("\n=== Comparison Complete ===")


if __name__ == '__main__':
    main()

