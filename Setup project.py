"""
Project structure setup for Telecom Tower RL Thesis
Run this in your MYPROJ/ directory to create all needed folders
"""

import os
from pathlib import Path

def setup_project_structure():
    """Create complete folder structure matching architecture doc"""
    
    # Root directory (run this from MYPROJ/)
    base = Path.cwd()
    
    # Define all folders
    folders = [
        # Data
        "data/raw",
        "data/processed",
        
        # Source code
        "src/env",
        "src/models",
        "src/train",
        "src/eval",
        "src/baselines",
        "src/utils",
        
        # Configs (already exists, but add subdirs)
        "configs/sites",
        "configs/scenarios",
        
        # Notebooks (already exists)
        "notebooks/eda",
        "notebooks/debug",
        
        # Results (already exists, but add subdirs)
        "results/checkpoints",
        "results/models",
        "results/eval",
        "results/figures",
        "results/logs",
        
        # Tests (important!)
        "tests/env",
        "tests/models",
    ]
    
    print("Creating project structure for Telecom Tower RL...")
    print(f"Base directory: {base}")
    print()
    
    created = []
    existed = []
    
    for folder in folders:
        path = base / folder
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(folder)
            print(f"✓ Created: {folder}")
        else:
            existed.append(folder)
            print(f"○ Exists:  {folder}")
    
    print()
    print(f"Summary: {len(created)} created, {len(existed)} existed")
    
    # Create __init__.py files for Python packages
    print()
    print("Creating __init__.py files...")
    
    init_dirs = [
        "src",
        "src/env",
        "src/models", 
        "src/train",
        "src/eval",
        "src/baselines",
        "src/utils",
        "tests",
        "tests/env",
        "tests/models",
    ]
    
    for dir in init_dirs:
        init_file = base / dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Auto-generated __init__.py\n")
            print(f"✓ Created: {dir}/__init__.py")
    
    print()
    print("✓ Project structure ready!")
    print()
    print("Next steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Start with notebooks/eda/01_dataset_exploration.ipynb")
    print("3. Then: src/env/telecom_env.py")

if __name__ == "__main__":
    setup_project_structure()