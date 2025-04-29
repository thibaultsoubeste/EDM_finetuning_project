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
    """Preprocess an image for the NIMA model."""
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

def score_image(model, img_tensor):
    """Score an image using the NIMA model."""
    with torch.no_grad():
        mean_t, dist_t = model(img_tensor)
    
    dist_np = dist_t.squeeze(0).cpu().numpy()
    mean = mean_t.item()
    std = np.sqrt(((np.arange(1, 11) - mean)**2 * dist_np).sum())
    
    return mean, std, dist_np

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
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level=log_level, log_file=args.log_file)
    
    logger.info(f"Starting batch image scoring")
    start_time = time.time()
    
    try:
        # Create the model
        model = create_nima_model(logger)
        
        # Get list of images to process
        if os.path.isdir(args.input):
            # Recursively find all image files in the directory and subdirectories
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
                image_paths.extend(glob.glob(os.path.join(args.input, '**', ext), recursive=True))
        else:
            # Handle glob patterns
            image_paths = glob.glob(args.input)
            
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process images
        results = []
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            img_tensor = preprocess_image(image_path, logger)
            if img_tensor is None:
                logger.warning(f"Skipping {image_path} due to processing error")
                continue
                
            mean_score, std_score, _ = score_image(model, img_tensor)
            
            results.append({
                'image_path': image_path,
                'mean_score': mean_score,
                'std_score': std_score
            })
            
        # Create results dataframe
        df = pd.DataFrame(results)
        
        # Save results to CSV
        os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.', exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        logger.info(f"Results saved to {args.output_csv}")
        
        # Print summary
        logger.info(f"\nScored {len(df)} images")
        logger.info(f"Mean aesthetic score: {df['mean_score'].mean():.2f}")
        logger.info(f"Median aesthetic score: {df['mean_score'].median():.2f}")
        logger.info(f"Min score: {df['mean_score'].min():.2f}, Max score: {df['mean_score'].max():.2f}")
        
        # Create distribution visualization
        if len(df) > 0:
            visualize_score_distribution(df['mean_score'].values, args.output_dist, logger)
        
        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        return 0
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())