from pathlib import Path
import os

# Optional: color codes for terminal
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# === Setup paths ===
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)  # Optional: ensures relative paths start from repo root

CODE_DIR = REPO_ROOT / 'ode_net'
DATA_DIR = REPO_ROOT / 'breast_cancer_data' / 'clean_data'
OUTPUT_DIR = REPO_ROOT / 'output'

settings_file = CODE_DIR / 'config_breast.cfg'
train_data_file = DATA_DIR / 'desmedt_500genes_1sample_178T.csv'
test_data_file = DATA_DIR / 'desmedt_500genes_1TESTsample_8middleT.csv'

# === Confirm directories ===
print("\n=== Checking directories ===")
for dir_path in [REPO_ROOT, CODE_DIR, DATA_DIR, OUTPUT_DIR]:
    abs_path = dir_path.resolve()
    if dir_path.exists():
        print(f"{GREEN}✅ Exists: {abs_path}{RESET}")
    else:
        print(f"{RED}❌ Missing: {abs_path}. Creating...{RESET}")
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"{GREEN}🟢 Created successfully: {abs_path}{RESET}")

# === Confirm files ===
print("\n=== Checking files ===")
for file_path in [settings_file, train_data_file, test_data_file]:
    abs_path = file_path.resolve()
    if file_path.exists():
        print(f"{GREEN}✅ Exists: {abs_path}{RESET}")
    else:
        print(f"{RED}❌ Missing: {abs_path}{RESET}")

# === Current working directory ===
print(f"\nCurrent working directory: {Path.cwd().resolve()}")
