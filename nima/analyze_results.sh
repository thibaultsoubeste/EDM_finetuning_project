#!/bin/bash
#SBATCH --job-name=NIMA_Score
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH -o result/outputs/nima_output_%j.txt
#SBATCH -e result/errors/nima_error_%j.txt
#SBATCH --time=0-23:59:00

# Check GPU status
nvidia-smi

# Load environment
source /nfs/sloanlab007/projects/diffusion_mban_proj/venvTibo/bin/activate

# Activate your conda environment if needed
# source activate your_env_name
# Set up paths
RESULTS_DIR="/nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_results"
OUTPUT_DIR="/nfs/sloanlab007/projects/diffusion_mban_proj/nima/analysis_results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Create a Python script for this specific job
cat > ${OUTPUT_DIR}/run_analysis.py << 'EOL'
import pandas as pd
import os
import glob
import numpy as np

def calculate_model_mean_scores(base_dir):
    """
    Calculate the mean score for each model based on CSV files.
    """
    # Dictionary to store results
    results = {
        "model_name": [],
        "mean_score": [],
        "std_error": [],
        "num_samples": []
    }
    
    print(f"Looking for CSV files in: {base_dir}")
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    # Identify unique model names
    model_names = set()
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # Extract model name from filename
        if "_checkpoint" in filename:
            continue  # Skip checkpoint files for now
        
        model_name = filename.replace(".csv", "")
        if model_name not in ["scores", "scores_all_1k"]:  # Skip aggregated files
            model_names.add(model_name)
    
    print(f"Found models: {model_names}")
    
    for model in model_names:
        # Get the CSV file for this model
        model_csv = os.path.join(base_dir, f"{model}.csv")
        
        if not os.path.exists(model_csv):
            print(f"CSV file not found for model: {model}")
            continue
            
        try:
            # Read CSV file
            df = pd.read_csv(model_csv)
            
            # Get mean scores column
            if 'mean_score' in df.columns:
                scores = df['mean_score'].values
                
                # Calculate statistics
                overall_mean = np.mean(scores)
                std_error = np.std(scores) / np.sqrt(len(scores))
                
                # Store results
                results["model_name"].append(model)
                results["mean_score"].append(overall_mean)
                results["std_error"].append(std_error)
                results["num_samples"].append(len(scores))
                
                print(f"Processed {model}: mean={overall_mean:.4f}, samples={len(scores)}")
            else:
                print(f"No 'mean_score' column found in {model_csv}")
        except Exception as e:
            print(f"Error processing {model_csv}: {e}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Format for table presentation
    formatted_df = results_df.copy()
    formatted_df["mean_score"] = formatted_df["mean_score"].round(4)
    formatted_df["std_error"] = formatted_df["std_error"].round(4)
    
    # Sort by mean score (descending)
    formatted_df = formatted_df.sort_values("mean_score", ascending=False)
    
    return formatted_df

def create_latex_table(df):
    """Convert DataFrame to LaTeX table for academic papers"""
    latex_code = "\\begin{table}[h]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{lccc}\n"
    latex_code += "\\hline\n"
    latex_code += "Model & Mean Score & Std Error & Samples \\\\ \\hline\n"
    
    for _, row in df.iterrows():
        model_name = row["model_name"].replace("_", "\\_")  # Escape underscores for LaTeX
        latex_code += f"{model_name} & {row['mean_score']:.4f} & {row['std_error']:.4f} & {row['num_samples']} \\\\\n"
    
    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\caption{Performance comparison of fully-trained vs. finetuned diffusion models}\n"
    latex_code += "\\label{tab:model_performance}\n"
    latex_code += "\\end{table}"
    
    return latex_code

def main():
    # Hardcoded paths instead of environment variables
    results_dir = "/nfs/sloanlab007/projects/diffusion_mban_proj/nima/nima_results"
    output_dir = "/nfs/sloanlab007/projects/diffusion_mban_proj/analysis_results"
    
    print(f"Analyzing model results in: {results_dir}")
    print(f"Saving output to: {output_dir}")
    
    # Calculate scores
    results_df = calculate_model_mean_scores(results_dir)
    
    # Display results
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    
    # Generate LaTeX table for paper
    latex_table = create_latex_table(results_df)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "model_performance_results.csv"), index=False)
    
    with open(os.path.join(output_dir, "model_performance_table.tex"), "w") as f:
        f.write(latex_table)
    
    print("\nResults saved to model_performance_results.csv")
    print("LaTeX table saved to model_performance_table.tex")

if __name__ == "__main__":
    main()
EOL

# Run the analysis
cd $OUTPUT_DIR
python run_analysis.py

echo "Analysis completed. Results are in $OUTPUT_DIR"