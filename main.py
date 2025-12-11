"""
AMR PATTERN RECOGNITION - MAIN PIPELINE

This is the main entry point for running the complete AMR analysis pipeline.

Phases:
    0. Data Understanding & Preparation
    1. Unsupervised Pattern Recognition
    2. Supervised Pattern Recognition
    3. Model Comparison & Interpretation
    4. Deployment Preparation
    5. Documentation & Reporting

Usage:
    python main.py
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.phase0_data_preparation import run_phase0
from src.phase1_unsupervised import run_phase1
from src.phase2_supervised import run_phase2
from src.phase3_comparison import run_phase3
from src.phase4_deployment import run_phase4
from src.phase5_documentation import run_phase5


def run_complete_pipeline(data_path: str = 'rawdata.csv',
                          output_dir: str = 'outputs',
                          phases: list = None) -> dict:
    """
    Run the complete AMR pattern recognition pipeline.
    
    Args:
        data_path: Path to the raw data CSV file
        output_dir: Directory to save all outputs
        phases: List of phases to run (e.g., [0, 1, 2, 3, 4, 5]). 
                If None, runs all phases.
    
    Returns:
        Dictionary containing results from all phases
    """
    # Default to all phases
    if phases is None:
        phases = [0, 1, 2, 3, 4, 5]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("AMR PATTERN RECOGNITION PIPELINE")
    print("=" * 70)
    print(f"\nData path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Phases to run: {phases}")
    print("=" * 70 + "\n")
    
    results = {}
    
    # Phase 0: Data Understanding & Preparation
    if 0 in phases:
        print("\n" + "▶" * 30)
        print("RUNNING PHASE 0")
        print("▶" * 30)
        results['phase0'] = run_phase0(data_path, output_dir)
    
    # Phase 1: Unsupervised Pattern Recognition
    if 1 in phases and 'phase0' in results:
        print("\n" + "▶" * 30)
        print("RUNNING PHASE 1")
        print("▶" * 30)
        processed_data = results['phase0']['mar_splits']['full_processed']
        feature_cols = results['phase0']['mar_splits']['feature_cols']
        results['phase1'] = run_phase1(processed_data, feature_cols, output_dir)
    
    # Phase 2: Supervised Pattern Recognition
    if 2 in phases and 'phase0' in results:
        print("\n" + "▶" * 30)
        print("RUNNING PHASE 2")
        print("▶" * 30)
        results['phase2'] = run_phase2(
            results['phase0']['mar_splits'],
            results['phase0']['species_splits'],
            output_dir
        )
    
    # Phase 3: Model Comparison & Interpretation
    if 3 in phases and 'phase2' in results:
        print("\n" + "▶" * 30)
        print("RUNNING PHASE 3")
        print("▶" * 30)
        results['phase3'] = run_phase3(
            results['phase2']['mar_prediction'],
            results['phase2']['species_classification'],
            output_dir
        )
    
    # Phase 4: Deployment Preparation
    if 4 in phases:
        print("\n" + "▶" * 30)
        print("RUNNING PHASE 4")
        print("▶" * 30)
        results['phase4'] = run_phase4(output_dir)
    
    # Phase 5: Documentation & Reporting
    if 5 in phases:
        print("\n" + "▶" * 30)
        print("RUNNING PHASE 5")
        print("▶" * 30)
        
        stats = results.get('phase0', {}).get('stats')
        phase1_results = results.get('phase1')
        phase2_results = results.get('phase2')
        
        results['phase5'] = run_phase5(stats, phase1_results, phase2_results, output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}/")
    print("\nGenerated files:")
    
    for f in sorted(Path(output_dir).glob('**/*')):
        if f.is_file():
            print(f"  - {f.relative_to(output_dir)}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AMR Pattern Recognition Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all phases
    python main.py
    
    # Run specific phases
    python main.py --phases 0 1 2
    
    # Use custom data file
    python main.py --data mydata.csv
    
    # Custom output directory
    python main.py --output results/
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='rawdata.csv',
        help='Path to the raw data CSV file (default: rawdata.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    parser.add_argument(
        '--phases', '-p',
        type=int,
        nargs='+',
        default=None,
        help='Phases to run (0-5). If not specified, runs all phases.'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    run_complete_pipeline(
        data_path=args.data,
        output_dir=args.output,
        phases=args.phases
    )
