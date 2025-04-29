#!/bin/bash
#SBATCH --job-name=NIMA_Score
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH -o nima_result/outputs/nima_output_%j.txt
#SBATCH -e nima_result/errors/nima_error_%j.txt
#SBATCH --time=0-23:59:00

# Check GPU status
nvidia-smi

# Load environment
source /nfs/sloanlab007/projects/diffusion_mban_proj/venvTibo/bin/activate

# Create directories for results and logs if they don't exist
mkdir -p /nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_results
mkdir -p /nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_logs

# Process images from a single class folder
python /nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_batch_scorer_percentage.py \
  --input "/nfs/sloanlab007/projects/diffusion_mban_proj/out/img_generated/xxs-lr1e-3-0025493" \
  --output-csv "/nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_results/xxs-lr1e-3-0025493.csv" \
  --output-dist "/nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_results/xxs-lr1e-3-0025493.png" \
  --log-file "/nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_logs/xxs-lr1e-3-0025493.log" \
  --log-level INFO \
  --sample-rate 1 \
  --checkpoint-interval 1000
  --batch-size 32

echo "Job completed at: $(date)"