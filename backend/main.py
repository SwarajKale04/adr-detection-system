"""
ADR Detection System - Main Entry Point.

This script orchestrates the Adverse Drug Reaction (ADR) detection pipeline.
It loads datasets, processes drug reviews, detects ADRs, validates them against
medical reference data, and outputs structured results.

Usage:
    python main.py
    
Expected Datasets:
    - data/DS1_WebMD_Reviews.xlsx: Drug reviews from WebMD
    - data/DS2_IndianSubstitutes.xlsx: Mapping of drugs to Indian equivalents
    - data/DS3_MedicineDataset.csv: Medical reference dataset
    
Output:
    - output/results.json: Structured ADR detection results
"""

import json
import logging
from pathlib import Path
from modules.pipeline import ADRDetectionPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_configuration() -> dict:
    """
    Load pipeline configuration.
    
    Returns:
        dict: Configuration dictionary with file paths and parameters
    """
    config = {
        'webmd_reviews_path': 'data/DS1_WebMD_Reviews.xlsx',
        'indian_substitutes_path': 'data/DS2_IndianSubstitutes.xlsx',
        'medicine_dataset_path': 'data/DS3_MedicineDataset.csv',
        'output_path': 'output/results.json',
        'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english',
        'confidence_threshold': 0.6,
        'batch_size': 32,
    }
    return config


def validate_input_files(config: dict) -> bool:
    """
    Validate that all required input files exist.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if all files exist, False otherwise
    """
    required_files = [
        config['webmd_reviews_path'],
        config['indian_substitutes_path'],
        config['medicine_dataset_path'],
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file not found: {file_path}")
            return False
    
    return True


def main():
    """
    Main execution function for ADR Detection System.
    
    Workflow:
    1. Load configuration
    2. Validate input files
    3. Initialize pipeline
    4. Run detection pipeline
    5. Save results
    6. Print summary
    """
    logger.info("=" * 60)
    logger.info("ADR Detection System - Starting")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_configuration()
    logger.info(f"Configuration loaded: {len(config)} parameters")
    
    # Validate input files
    if not validate_input_files(config):
        logger.error("Input file validation failed. Exiting.")
        return False
    
    logger.info("Input files validated successfully")
    
    # Initialize and run pipeline
    try:
        pipeline = ADRDetectionPipeline()
        logger.info("Pipeline initialized")
        
        # Run detection pipeline
        logger.info("Starting ADR detection pipeline...")
        results = pipeline.run(config)
        
        # Get execution summary
        summary = pipeline.get_execution_summary()
        logger.info(f"Pipeline execution completed")
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")
        
        # Save results
        pipeline.save_results(config['output_path'])
        logger.info(f"Results saved to: {config['output_path']}")
        
        logger.info("=" * 60)
        logger.info("ADR Detection System - Completed Successfully")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
