#!/usr/bin/env python3
"""
find_missing_files.py
----------------------------------
Scans all Python scripts in a directory, extracts file references
(open, pandas.read_*, numpy.load, etc.), and reports which ones
are missing from the filesystem.

Outputs:
  • missing_files_report.txt  – human-readable summary
  • missing_files_report.json – structured data (for analysis or HTML export)

Usage:
    python find_missing_files.py [optional_path]

Dependencies:
    pip install tqdm
"""

import os
import ast
import sys
import json
from pathlib import Path
from tqdm import tqdm


# ---------- 1. Collect all files ----------
def list_all_files(root_dir: str) -> set[str]:
    all_files = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            all_files.add(os.path.relpath(os.path.join(dirpath, f), root_dir))
    return all_files


# ---------- 2. Extract file references ----------
TARGET_FUNCS = {
    "open",
    "read_csv",
    "read_excel",
    "read_json",
    "read_table",
    "load",
    "save",
    "to_csv",
    "to_excel",
    "to_json",
    "np.load",
    "pd.read_csv",
    "pd.read_excel",
    "pd.read_json",
}


def extract_file_references_with_lines(py_file: Path):
    """Return list of (referenced_path, line_number) from given Python file."""
    refs = []
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))
    except (SyntaxError, UnicodeDecodeError):
        return refs

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"

            if func_name in TARGET_FUNCS and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    refs.append((first_arg.value, node.lineno))
                elif isinstance(first_arg, ast.JoinedStr):
                    for v in first_arg.values:
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            refs.append((v.value, node.lineno))
    return refs


# ---------- 3. Main logic ----------
def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    root_dir = os.path.abspath(root_dir)

    print(f"📂 Scanning project directory: {root_dir}\n")

    existing_files = list_all_files(root_dir)
    referenced_files = {}  # ref_path -> list of (script, line)

    py_files = list(Path(root_dir).rglob("*.py"))

    for py in tqdm(py_files, desc="🔍 Scanning Python files", unit="file"):
        for ref, lineno in extract_file_references_with_lines(py):
            referenced_files.setdefault(ref, []).append((str(py), lineno))

    missing = {
        ref: locations
        for ref, locations in referenced_files.items()
        if not any(Path(root_dir, f).name == Path(ref).name for f in existing_files)
    }

    print("\n✅ Existing file count:", len(existing_files))
    print("📑 Referenced file count:", len(referenced_files))
    print()

    # ---------- Write text report ----------
    txt_path = "missing_files_report.txt"
    with open(txt_path, "w", encoding="utf-8") as out:
        if missing:
            out.write("Missing file references (with line numbers):\n\n")
            for ref, locations in sorted(missing.items()):
                out.write(f"{ref}\n")
                for script, line in locations:
                    out.write(f"    {script}:{line}\n")
        else:
            out.write("No missing file references detected.\n")

    # ---------- Write JSON report ----------
    json_path = "missing_files_report.json"
    json_data = {
        "root_directory": root_dir,
        "existing_file_count": len(existing_files),
        "referenced_file_count": len(referenced_files),
        "missing_files": {
            ref: [{"script": s, "line": l} for s, l in locs]
            for ref, locs in sorted(missing.items())
        },
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_data, jf, indent=2)

    print(f"📝 Reports saved:")
    print(f"   • {txt_path}")
    print(f"   • {json_path}")


if __name__ == "__main__":
    main()
