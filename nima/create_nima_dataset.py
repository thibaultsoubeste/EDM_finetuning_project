#!/usr/bin/env python3
import os
import csv
import shutil
import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def setup_logging(log_dir):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"filter_ilsvrc_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configured. Log file: {log_file}")
    return log_file

def parse_arguments():
    parser = argparse.ArgumentParser(description='Filter ImageNet dataset based on CSV score file')
    parser.add_argument('--csv_path', type=str, required=True, 
                        help='Path to the CSV file with scores and image paths')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to the original train directory (containing class folders)')
    parser.add_argument('--target_dir', type=str, required=True, 
                        help='Path to create the filtered dataset directory')
    parser.add_argument('--path_column', type=str, default='image_path',
                        help='Name of the column in CSV containing image paths')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to store log files')
    parser.add_argument('--results_csv', type=str, default=None,
                        help='Path to save results CSV with processed files info')
    parser.add_argument('--score_threshold', type=float, default=None,
                        help='Optional threshold for NIMA score to filter images')
    parser.add_argument('--score_column', type=str, default='mean_score',
                        help='Name of the column containing the NIMA score')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set up logging
    log_file = setup_logging(args.log_dir)
    
    # Create target directory structure if it doesn't exist
    target_train_dir = os.path.join(args.target_dir, 'CLS-LOC', 'train')
    os.makedirs(target_train_dir, exist_ok=True)
    logging.info(f"Created target directory: {target_train_dir}")
    
    # Read paths from CSV file
    logging.info(f"Reading data from {args.csv_path}")
    try:
        df = pd.read_csv(args.csv_path)
        if args.path_column not in df.columns:
            logging.error(f"Column '{args.path_column}' not found in CSV. Available columns: {df.columns.tolist()}")
            return
            
        # Apply score threshold if specified
        if args.score_threshold is not None:
            if args.score_column not in df.columns:
                logging.error(f"Score column '{args.score_column}' not found. Available columns: {df.columns.tolist()}")
                return
            
            original_count = len(df)
            df = df[df[args.score_column] >= args.score_threshold]
            logging.info(f"Applied score threshold {args.score_threshold}: kept {len(df)} out of {original_count} images")
        
        paths_from_csv = df[args.path_column].tolist()
        logging.info(f"Found {len(paths_from_csv)} paths in the CSV file to process")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return
    
    # Prepare results tracking
    results = {
        'path': [],
        'class_folder': [],
        'status': [],
        'message': []
    }
    
    # Process each path
    copied_files = 0
    skipped_files = 0
    error_files = 0
    
    logging.info("Starting to process files...")
    
    # Use tqdm for progress tracking
    for img_path in tqdm(paths_from_csv, desc="Processing images"):
        try:
            # Extract the class folder and image file from the path
            path_parts = img_path.split('/')
            
            # For paths like /nfs/.../imagenet21k_train/n02835915/n02835915_13889.JPEG
            # The class folder is the second-to-last element
            # The image file is the last element
            class_folder = path_parts[-2]  # e.g., n02835915
            image_file = path_parts[-1]    # e.g., n02835915_13889.JPEG
            
            # Check if the file exists in the source directory
            full_source_path = os.path.join(args.source_dir, class_folder, image_file)
            
            # Check if file exists
            if os.path.exists(full_source_path) and os.path.isfile(full_source_path):
                # Create target directory structure
                target_file = os.path.join(target_train_dir, class_folder, image_file)
                target_dir = os.path.dirname(target_file)
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy file
                try:
                    shutil.copy2(full_source_path, target_file)
                    copied_files += 1
                    if copied_files % 1000 == 0:
                        logging.info(f"Processed {copied_files} files...")
                    
                    # Log success
                    results['path'].append(img_path)
                    results['class_folder'].append(class_folder)
                    results['status'].append('copied')
                    results['message'].append(f"Copied to {target_file}")
                    
                except Exception as e:
                    error_message = f"Error copying {full_source_path} to {target_file}: {e}"
                    logging.error(error_message)
                    error_files += 1
                    
                    # Log error
                    results['path'].append(img_path)
                    results['class_folder'].append(class_folder)
                    results['status'].append('error')
                    results['message'].append(error_message)
            else:
                skipped_files += 1
                
                # Log skipped
                results['path'].append(img_path)
                results['class_folder'].append(class_folder if 'class_folder' in locals() else 'unknown')
                results['status'].append('skipped')
                results['message'].append(f"File not found: {full_source_path}")
        
        except Exception as e:
            error_message = f"Error processing path {img_path}: {e}"
            logging.error(error_message)
            error_files += 1
            
            # Log general processing error
            results['path'].append(img_path)
            results['class_folder'].append('error')
            results['status'].append('error')
            results['message'].append(error_message)
    
    # Log summary
    summary = f"Finished processing. Copied {copied_files} files, skipped {skipped_files} files, errors on {error_files} files."
    logging.info(summary)
    
    # Save results CSV if requested
    if args.results_csv:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.results_csv, index=False)
        logging.info(f"Results saved to {args.results_csv}")
    
    # Generate summary files
    summary_file = os.path.join(args.log_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CSV file: {args.csv_path}\n")
        f.write(f"Source directory: {args.source_dir}\n")
        f.write(f"Target directory: {args.target_dir}\n")
        f.write(f"Log file: {log_file}\n")
        f.write(f"Total paths in CSV: {len(paths_from_csv)}\n")
        f.write(f"Files copied: {copied_files}\n")
        f.write(f"Files skipped: {skipped_files}\n")
        f.write(f"Files with errors: {error_files}\n")
        
        if args.score_threshold is not None:
            f.write(f"Applied NIMA score threshold: {args.score_threshold}\n")
    
    # Create a file with successful class folders for verification
    class_dirs = set()
    for class_folder in results['class_folder']:
        if class_folder != 'error' and class_folder != 'unknown':
            class_dirs.add(class_folder)
    
    with open(os.path.join(args.log_dir, "classes_created.txt"), 'w') as f:
        for class_dir in sorted(class_dirs):
            f.write(f"{class_dir}\n")
    
    logging.info(f"Summary saved to {summary_file}")
    logging.info(f"Created {len(class_dirs)} class directories")

if __name__ == "__main__":
    main()