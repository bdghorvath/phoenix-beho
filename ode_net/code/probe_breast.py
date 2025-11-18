import itertools
import subprocess
import pprint
import sys
from pathlib import Path

print("🚀 Starting hyperparameter grid search...")

# -------------------------------------------------------------------------
# ❗️ EDIT THIS: Set the name for your new experiment's parent folder
# This will be created inside your main 'output' directory.
# This keeps it separate from your last experiment.
GRID_SEARCH_NAME = "my_awesome_grid_search_v1"
# -------------------------------------------------------------------------

# This script assumes your main output folder is named 'output'
# as defined in train_breast_probing.py
MAIN_OUTPUT_DIR = Path("output")
GRID_SEARCH_ROOT = MAIN_OUTPUT_DIR / GRID_SEARCH_NAME


# 1. Define the base (static) settings
base_config = {
    "viz": True,
    "method": "dopri5",
    "dec_lr": False,
    "dec_lr_factor": 0.995,
    "cpu": True,
    "val_split": 0.04,
    "noise": 0,
    "epochs": 100,
    "normalize_data": False,
    "batch_type": "single",
    "pretrained_model": False,
    "log_scale": "linear",
    "scale_expression": 1.0,
    "explicit_time": False,
    "relative_error": False,
    "batch_time": 10,
    "batch_time_frac": 0.5,
    "debug": False,
    "verbose": True,
}

# 2. Define the search space (iterated settings)
search_space = {
    'neurons_per_layer': [40, 60, 120],
    'optimizer': ['adam', 'adamw', 'sgd'],
    'batch_size': [17, 16, 32],
    'init_lr': [5e-4, 1e-3, 1e-4],
    'weight_decay': [0.0, 1e-5, 1e-4]
}

# 3. Prepare for iteration
param_keys = list(search_space.keys())
param_values = list(search_space.values())
all_combinations = list(itertools.product(*param_values))
total_combinations = len(all_combinations)

print(f"All outputs will be saved in separate directories inside: {GRID_SEARCH_ROOT.resolve()}")
print(f"Total experiments to run: {total_combinations}\n")

# 4. Iterate through all combinations
for i, combination in enumerate(all_combinations):
    
    print(f"--- Experiment {i+1} / {total_combinations} ---")
    
    # Start with the base config
    current_config = base_config.copy()
    
    # Create a dict for the current iteration's settings
    combination_dict = dict(zip(param_keys, combination))
    
    # Update the base config with the iteration-specific settings
    current_config.update(combination_dict)

    # --- NEW: Generate Unique Directory for this run ---
    
    # Create a descriptive name, e.g.: "npl_40_opt_adam_bs_17_lr_0.0005_wd_0.0"
    run_name = (
        f"npl_{combination_dict['neurons_per_layer']}"
        f"_opt_{combination_dict['optimizer']}"
        f"_bs_{combination_dict['batch_size']}"
        f"_lr_{combination_dict['init_lr']}"
        f"_wd_{combination_dict['weight_decay']}"
    )
    
    # Create the full path for this specific run's output
    # e.g., output/my_awesome_grid_search_v1/npl_40_...
    run_output_path = GRID_SEARCH_ROOT / run_name
    
    # Add this new path to the config we will pass
    current_config['run_output_dir'] = str(run_output_path)
    
    # ----------------------------------------------------

    print("Running with configuration:")
    pprint.pprint(current_config)

    # 5. Build the command-line command
    command = ['python', 'train_breast_probing.py']
    
    for key, value in current_config.items():
        command.append(f'--{key}')
        command.append(str(value))
        
    # 6. Run the experiment
    try:
        subprocess.run(command, check=True)
        print(f"✅ Experiment {i+1} completed successfully. Output in: {run_output_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {i+1} FAILED: {e} ❌\n")
    except KeyboardInterrupt:
        print("\n🛑 User interrupted. Stopping grid search.")
        sys.exit(0)

print("🎉 Grid Search complete.")