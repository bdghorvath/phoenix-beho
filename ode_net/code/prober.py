import os
import subprocess
from datetime import datetime
from pathlib import Path

# === CONFIGURATION ===
# 1. Setup the list of learning rates you want to test
learning_rates = [0.1, 0.01, 0.005, 0.001, 0.0001]

# 2. Define your dataset name (Level 1 Folder)
dataset_name = "desmedt_500genes"

# 3. Define the path to your training script
repo_root = Path.cwd().parent.parent # Adjust based on where you save this file
script_path = "train_breast.py" # Assuming this is the name of your main script

# 4. Base results directory
base_results_dir = Path(repo_root) / "ode_net" / "code" / "output"

# === EXECUTION LOOP ===
for lr in learning_rates:
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d(%H;%M)")
    
    # FIX: Convert LR to string and replace '.' with '-' to prevent path errors
    # Example: 0.001 becomes "0-001"
    lr_str = str(lr).replace('.', '-')
    
    # Use the safe string in the folder name
    run_folder_name = f"{timestamp}_LR_{lr_str}"
    
    # Construct the full path
    full_output_path = base_results_dir / dataset_name / run_folder_name
    
    # Ensure directory exists
    os.makedirs(full_output_path, exist_ok=True)
    
    print(f"\n==================================================")
    print(f"STARTING RUN: LR={lr}")
    print(f"SAVING TO: {full_output_path}")
    print(f"==================================================\n")

    # Run the training script
    subprocess.run([
        "python", script_path,
        "--lr", str(lr),
        "--forced_out_dir", str(full_output_path)
    ])

print("\nAll experiments completed.")