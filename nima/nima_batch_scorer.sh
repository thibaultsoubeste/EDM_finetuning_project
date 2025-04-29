#!/bin/bash
#SBATCH --job-name=NIMA_Score
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH -o result/outputs/nima_output_%j.txt
#SBATCH -e result/errors/nima_error_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jordancn@mit.edu
#SBATCH --time=0-23:59:00

# Check GPU status
nvidia-smi

# Load environment
source /nfs/sloanlab007/projects/diffusion_mban_proj/venvTibo/bin/activate

pip install pandas
pip install glob

# Create directories for results and logs if they don't exist
mkdir -p /nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_results
mkdir -p /nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_logs

# Process images from a single class folder
python /nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_batch_scorer.py \
  --input "/nfs/sloanlab007/projects/diffusion_mban_proj/imagenet21k_resized/imagenet21k_train" \
  --output-csv "/nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_results/scores.csv" \
  --output-dist "/nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_results/score_distribution.png" \
  --log-file "/nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_logs/nima_batch.log" \
  --log-level INFO

echo "Job completed at: $(date)"
