#!/bin/bash
#SBATCH --job-name=NIMA_Score
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:59:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jordancn@mit.edu

# Create output directories first
mkdir -p nima_result/outputs
mkdir -p nima_result/errors

# Then set the SLURM output and error files
#SBATCH -o nima_result/outputs/nima_output_%j.txt
#SBATCH -e nima_result/errors/nima_error_%j.txt

# Check GPU status
nvidia-smi

# Load environment
source /nfs/sloanlab007/projects/diffusion_mban_proj/venvTibo/bin/activate

# Print job information
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"

# Install required packages
echo "Installing required Python packages..."
pip install pandas pathlib tqdm

# Set paths and variables
CSV_PATH="/nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_results/scores.csv"
SOURCE_DIR="/nfs/sloanlab007/projects/diffusion_mban_proj/dataset/ILSVRC/Data/CLS-LOC/train"
TARGET_DIR="/nfs/sloanlab007/projects/diffusion_mban_proj/dataset/Data_nima"
PATH_COLUMN="image_path"  

# Create directories with verbose output
echo "Creating target directory: $TARGET_DIR"
mkdir -p $TARGET_DIR
if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Failed to create target directory $TARGET_DIR"
    exit 1
fi
echo "Target directory created successfully"

# Create log directory with verbose output
LOG_DIR="${TARGET_DIR}/logs"
echo "Creating log directory: $LOG_DIR"
mkdir -p $LOG_DIR
if [ ! -d "$LOG_DIR" ]; then
    echo "ERROR: Failed to create log directory $LOG_DIR"
    # Fall back to current directory for logs
    LOG_DIR="."
    echo "Falling back to current directory for logs"
fi
echo "Log directory created/verified successfully"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_CSV="${LOG_DIR}/results_${TIMESTAMP}.csv"

# Save environment information with error checking
ENV_INFO="${LOG_DIR}/env_info_${TIMESTAMP}.txt"
echo "Saving environment info to: $ENV_INFO"
{
    echo "Environment information:"
    echo "Python version: $(python --version 2>&1)"
    echo "Pip version: $(pip --version 2>&1)"
    echo "Working directory: $(pwd)"
    echo "Slurm job ID: $SLURM_JOB_ID"
    echo "Node: $(hostname)"
    echo "CSV path: $CSV_PATH"
    echo "Source directory: $SOURCE_DIR"
    echo "Target directory: $TARGET_DIR"
    echo "Log directory: $LOG_DIR"
    echo ""
    echo "Installed packages:"
    pip list
} > "$ENV_INFO" 2>&1

# Verify critical paths exist
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV file not found at $CSV_PATH"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found at $SOURCE_DIR"
    exit 1
fi

# Run the Python script with timing
echo "Starting Python script at $(date)"
time python create_nima_dataset.py \
    --csv_path "$CSV_PATH" \
    --source_dir "$SOURCE_DIR" \
    --target_dir "$TARGET_DIR" \
    --path_column "$PATH_COLUMN" \
    --log_dir "$LOG_DIR" \
    --results_csv "$RESULTS_CSV"

# Check if the Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Python script completed successfully at $(date)"
else
    echo "Python script failed with exit code $? at $(date)"
    exit 1
fi

echo "Job completed!"