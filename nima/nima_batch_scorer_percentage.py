import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import logging
import sys
import time
import random  # Added for random sampling
from datetime import datetime
import glob
import pandas as pd

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_logger(log_level=logging.INFO, log_file=None):
    """Set up and configure logger."""
    logger = logging.getLogger('nima')
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_nima_model(logger):
    """Create a NIMA model using MobileNet as the base."""
    logger.info(f"Creating NIMA model with MobileNet base on {device}")
    
    class NIMAMobileNet(nn.Module):
        def __init__(self, bins=10):
            super().__init__()
            self.mnet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            self.mnet.classifier = nn.Sequential(  
                nn.Dropout(0.75),
                nn.Linear(self.mnet.last_channel, bins)
            )
        def forward(self, x):
            logits = self.mnet(x)
            prob = F.softmax(logits, dim=-1)
            mean = (prob * torch.arange(1, 11, device=prob.device)).sum(1)
            return mean, prob
    
    model = NIMAMobileNet().to(device).eval()
    return model

# Image preprocessing pipeline
imagenet_norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
preproc = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    imagenet_norm,
])

def preprocess_image(image_path, logger):
    """Preprocess a single image for the NIMA model."""
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        img = Image.open(image_path).convert("RGB")
        tensor = preproc(img).unsqueeze(0).to(device)
        return tensor
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None

def preprocess_batch(image_paths, logger, batch_size=16):
    """Preprocess a batch of images for the NIMA model."""
    valid_tensors = []
    valid_paths = []
    
    for path in image_paths:
        try:
            if not os.path.exists(path):
                logger.error(f"Image file not found: {path}")
                continue
            
            img = Image.open(path).convert("RGB")
            tensor = preproc(img).unsqueeze(0)
            valid_tensors.append(tensor)
            valid_paths.append(path)
            
        except Exception as e:
            logger.error(f"Error processing {path}: {str(e)}")
            continue
    
    if not valid_tensors:
        return None, []
    
    # Stack all tensors into a single batch
    batch_tensor = torch.cat(valid_tensors, dim=0).to(device)
    return batch_tensor, valid_paths

def score_image(model, img_tensor):
    """Score a single image using the NIMA model."""
    with torch.no_grad():
        mean_t, dist_t = model(img_tensor)
    
    dist_np = dist_t.squeeze(0).cpu().numpy()
    mean = mean_t.item()
    std = np.sqrt(((np.arange(1, 11) - mean)**2 * dist_np).sum())
    
    return mean, std, dist_np

def score_batch(model, img_tensors):
    """Score a batch of images using the NIMA model."""
    with torch.no_grad():
        mean_t, dist_t = model(img_tensors)
    
    # Process each image in the batch
    results = []
    for i in range(mean_t.shape[0]):
        mean = mean_t[i].item()
        dist_np = dist_t[i].cpu().numpy()
        std = np.sqrt(((np.arange(1, 11) - mean)**2 * dist_np).sum())
        results.append((mean, std, dist_np))
    
    return results

def visualize_score_distribution(scores, output_path, logger):
    """Create a histogram of all image scores."""
    logger.info("Creating score distribution visualization")
    
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(scores), color='r', linestyle='--', 
                label=f'Mean: {np.mean(scores):.2f}')
    plt.axvline(np.median(scores), color='g', linestyle='-', 
                label=f'Median: {np.median(scores):.2f}')
    plt.xlabel('Aesthetic Score')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Aesthetic Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Distribution saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Score multiple images using NIMA model')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to directory containing images or a glob pattern')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Path to save results CSV')
    parser.add_argument('--output-dist', type=str, required=True,
                        help='Path to save distribution visualization')
    parser.add_argument('--log-file', type=str, default='nima_batch.log',
                        help='Path to log file')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('--sample-rate', type=float, default=0.05,
                        help='Percentage of images to sample from each subfolder (0.0-1.0, default: 0.05 or 5%%)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Number of images to process before saving a checkpoint (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Number of images to process in a single batch (default: 16)')
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level=log_level, log_file=args.log_file)
    
    logger.info(f"Starting batch image scoring")
    logger.info(f"Using device: {device}")
    logger.info(f"Input path: {args.input}")
    logger.info(f"Output CSV: {args.output_csv}")
    logger.info(f"Output distribution: {args.output_dist}")
    start_time = time.time()
    
    try:
        # Create the model
        logger.info("Starting model creation...")
        model_start_time = time.time()
        model = create_nima_model(logger)
        model_end_time = time.time()
        logger.info(f"Model creation completed in {model_end_time - model_start_time:.2f} seconds")
        
        # Get list of images to process
        logger.info(f"Searching for images in: {args.input}")
        logger.info(f"Using sample rate: {args.sample_rate * 100:.1f}%")
        image_search_start = time.time()
        
        if os.path.isdir(args.input):
            # The structure is: main folder -> subfolders -> images
            # We need to sample from each subfolder
            image_paths = []
            subfolder_counts = {}
            
            # First, identify all subfolders
            subfolders = [f.path for f in os.scandir(args.input) if f.is_dir()]
            logger.info(f"Found {len(subfolders)} subfolders to sample from")
            
            # If no subfolders, treat the input directory as a single folder
            if len(subfolders) == 0:
                logger.info(f"No subfolders found, treating {args.input} as a single folder")
                subfolders = [args.input]
            
            # Process each subfolder
            for subfolder in subfolders:
                subfolder_images = []
                for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']:
                    # Find all files with this extension in the subfolder
                    for file in os.listdir(subfolder):
                        if file.endswith(ext):
                            subfolder_images.append(os.path.join(subfolder, file))
                
                # Log the number of images found in this subfolder
                logger.info(f"Found {len(subfolder_images)} images in subfolder {os.path.basename(subfolder)}")
                
                # Sample images from this subfolder based on the provided rate
                if len(subfolder_images) > 0:
                    sample_size = max(1, int(len(subfolder_images) * args.sample_rate))  # At least 1 image
                    sampled_images = random.sample(subfolder_images, min(sample_size, len(subfolder_images)))
                    image_paths.extend(sampled_images)
                    
                    subfolder_name = os.path.basename(subfolder)
                    subfolder_counts[subfolder_name] = {
                        'total': len(subfolder_images),
                        'sampled': sample_size
                    }
                    
                    logger.info(f"Subfolder '{subfolder_name}': Sampled {sample_size} of {len(subfolder_images)} images ({sample_size/len(subfolder_images)*100:.1f}%)")
                else:
                    logger.warning(f"Subfolder '{os.path.basename(subfolder)}': No images found")
            
            # Log sampling summary
            total_available = sum(info['total'] for info in subfolder_counts.values())
            total_sampled = sum(info['sampled'] for info in subfolder_counts.values())
            if total_available > 0:
                logger.info(f"Overall sampling: {total_sampled} of {total_available} images ({total_sampled/total_available*100:.1f}%)")
            else:
                logger.warning("No images found in any subfolders!")
        else:
            # Handle direct file pattern
            logger.info(f"Input is not a directory, treating as a file pattern: {args.input}")
            image_paths = glob.glob(args.input)
            
        image_search_end = time.time()
        logger.info(f"Image search completed in {image_search_end - image_search_start:.2f} seconds")
        logger.info(f"Found {len(image_paths)} images to process")
        
        if len(image_paths) == 0:
            logger.error(f"No images found to process! Please check the input path.")
            return 1
            
        # Log some example paths to verify correct discovery
        if len(image_paths) > 0:
            logger.info(f"Example image paths (first 5 or fewer):")
            for i, path in enumerate(image_paths[:5]):
                logger.info(f"  {i+1}. {path}")
        
        # Process images in batches
        results = []
        process_start_time = time.time()
        successful_images = 0
        failed_images = 0
        last_checkpoint = 0
        
        # Process images in batches
        for batch_idx in range(0, len(image_paths), args.batch_size):
            batch_start_time = time.time()
            
            # Get the current batch of images
            batch_paths = image_paths[batch_idx:batch_idx + args.batch_size]
            
            # Log progress
            if batch_idx % (args.batch_size * 10) == 0 or batch_idx + args.batch_size >= len(image_paths):
                elapsed = time.time() - process_start_time
                images_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                estimated_total = elapsed / (batch_idx + 1) * len(image_paths) if batch_idx > 0 else 0
                remaining = estimated_total - elapsed if batch_idx > 0 else 0
                
                logger.info(f"Progress: {batch_idx}/{len(image_paths)} images ({batch_idx/len(image_paths)*100:.1f}%)")
                logger.info(f"Speed: {images_per_sec:.2f} images/sec, Est. remaining: {remaining/60:.1f} minutes")
            
            try:
                # Preprocess the batch
                batch_tensor, valid_paths = preprocess_batch(batch_paths, logger, args.batch_size)
                
                if batch_tensor is None or len(valid_paths) == 0:
                    logger.warning(f"Skipping batch {batch_idx//args.batch_size} due to processing errors")
                    failed_images += len(batch_paths)
                    continue
                
                # Score the batch
                batch_results = score_batch(model, batch_tensor)
                
                # Add results to the list
                for path, (mean_score, std_score, _) in zip(valid_paths, batch_results):
                    results.append({
                        'image_path': path,
                        'mean_score': mean_score,
                        'std_score': std_score
                    })
                
                successful_images += len(valid_paths)
                failed_images += len(batch_paths) - len(valid_paths)
                
                # Save checkpoint periodically
                images_processed = batch_idx + len(batch_paths)
                if images_processed % args.checkpoint_interval < args.batch_size or batch_idx + args.batch_size >= len(image_paths):
                    checkpoint_start = time.time()
                    # Create checkpoint dataframe from new results only
                    checkpoint_df = pd.DataFrame(results[last_checkpoint:])
                    
                    # Create checkpoint filename
                    checkpoint_dir = os.path.dirname(args.output_csv)
                    base_name = os.path.basename(args.output_csv)
                    name, ext = os.path.splitext(base_name)
                    checkpoint_file = os.path.join(checkpoint_dir, f"{name}_checkpoint{ext}")
                    
                    # Save incremental results
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # If this is not the first checkpoint, append to existing file
                    if last_checkpoint > 0 and os.path.exists(checkpoint_file):
                        # Read existing checkpoint file
                        existing_df = pd.read_csv(checkpoint_file)
                        # Append new results
                        updated_df = pd.concat([existing_df, checkpoint_df], ignore_index=True)
                        # Save updated checkpoint
                        updated_df.to_csv(checkpoint_file, index=False)
                    else:
                        # Save first checkpoint
                        checkpoint_df.to_csv(checkpoint_file, index=False)
                    
                    last_checkpoint = len(results)
                    checkpoint_end = time.time()
                    logger.info(f"Saved checkpoint after {images_processed} images to {checkpoint_file} ({checkpoint_end - checkpoint_start:.2f} seconds)")
                
                batch_time = time.time() - batch_start_time
                logger.debug(f"Batch {batch_idx//args.batch_size} processed in {batch_time:.2f} seconds ({len(valid_paths)/batch_time:.2f} images/sec)")
                
            except Exception as e:
                logger.error(f"Error processing batch starting at index {batch_idx}: {str(e)}")
                failed_images += len(batch_paths)
                continue
            
        process_end_time = time.time()
        process_duration = process_end_time - process_start_time
        
        logger.info(f"Image processing complete")
        logger.info(f"Processed {successful_images} images successfully in {process_duration:.2f} seconds")
        logger.info(f"Failed to process {failed_images} images")
        if successful_images > 0:
            logger.info(f"Average processing time: {process_duration/successful_images:.4f} seconds per image")
        
        # Create results dataframe
        if len(results) == 0:
            logger.error("No images were processed successfully!")
            return 1
            
        df = pd.DataFrame(results)
        
        # Save results to CSV
        logger.info(f"Saving results to CSV: {args.output_csv}")
        csv_start_time = time.time()
        os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.', exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        csv_end_time = time.time()
        logger.info(f"Results saved to {args.output_csv} in {csv_end_time - csv_start_time:.2f} seconds")
        
        # Print summary
        logger.info(f"\nSCORE SUMMARY:")
        logger.info(f"Total images scored: {len(df)}")
        logger.info(f"Mean aesthetic score: {df['mean_score'].mean():.2f}")
        logger.info(f"Median aesthetic score: {df['mean_score'].median():.2f}")
        logger.info(f"Min score: {df['mean_score'].min():.2f}, Max score: {df['mean_score'].max():.2f}")
        logger.info(f"Standard deviation of scores: {df['mean_score'].std():.2f}")
        
        # Create distribution visualization
        if len(df) > 0:
            logger.info(f"Creating score distribution visualization: {args.output_dist}")
            viz_start_time = time.time()
            visualize_score_distribution(df['mean_score'].values, args.output_dist, logger)
            viz_end_time = time.time()
            logger.info(f"Visualization saved in {viz_end_time - viz_start_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"NIMA batch processing completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())