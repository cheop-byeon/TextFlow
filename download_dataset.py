#!/usr/bin/env python3
"""
Download TextFlow datasets from HuggingFace Hub to local storage.

This script downloads the TextFlow datasets from:
https://huggingface.co/datasets/jiebi/TextFlow

The datasets are stored in a local 'dataset' directory with the following structure:
- dataset/ids.i2c.dev.generation.jsonl
- dataset/ids.i2c.test.generation.jsonl
- dataset/ids.i2c.supplementary.generation.jsonl
"""

import os
import logging
from pathlib import Path
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configuration
REPO_ID = "jiebi/TextFlow"
DATASET_DIR = Path("dataset")
DATASET_FILES = {
    "train": "ids.i2c.train.generation.jsonl",
    "dev": "ids.i2c.dev.generation.jsonl",
    "test": "ids.i2c.test.generation.jsonl",
    "supp": "ids.i2c.supplementary.generation.jsonl",
}


def download_dataset(split: str = None) -> None:
    """
    Download datasets from HuggingFace Hub.
    
    Args:
        split: Specific split to download ('dev', 'test', 'supp'). 
               If None, downloads all splits.
    """
    # Create dataset directory if it doesn't exist
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Dataset directory: {DATASET_DIR.absolute()}")
    
    # Determine which splits to download
    splits_to_download = {split: DATASET_FILES[split]} if split else DATASET_FILES
    
    for split_name, filename in splits_to_download.items():
        output_path = DATASET_DIR / filename
        
        # Skip if file already exists
        if output_path.exists():
            logger.info(f"✓ {filename} already exists at {output_path}")
            continue
        
        logger.info(f"Downloading {split_name} split ({filename})...")
        
        try:
            # Load dataset from HuggingFace Hub
            dataset = load_dataset(REPO_ID, split=split_name)
            
            # Save to JSONL format
            dataset.to_json(str(output_path))
            
            logger.info(f"✓ Successfully saved {filename} ({len(dataset)} examples)")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
            raise


def verify_datasets() -> bool:
    """Verify that all required dataset files exist."""
    logger.info("Verifying datasets...")
    all_exist = True
    
    for split_name, filename in DATASET_FILES.items():
        output_path = DATASET_DIR / filename
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # Convert to MB
            logger.info(f"✓ {filename} ({file_size:.2f} MB)")
        else:
            logger.warning(f"✗ {filename} not found")
            all_exist = False
    
    return all_exist


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download TextFlow datasets from HuggingFace Hub"
    )
    parser.add_argument(
        "--split",
        choices=["train","dev", "test", "supp"],
        default=None,
        help="Specific split to download. If not specified, downloads all splits."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing datasets without downloading."
    )
    
    args = parser.parse_args()
    
    try:
        if args.verify_only:
            logger.info("Verification mode only")
            if verify_datasets():
                logger.info("✓ All datasets verified successfully!")
            else:
                logger.warning("⚠ Some datasets are missing")
                return 1
        else:
            download_dataset(split=args.split)
            
            if verify_datasets():
                logger.info("✓ All datasets downloaded and verified successfully!")
            else:
                logger.warning("⚠ Some datasets may be incomplete")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
