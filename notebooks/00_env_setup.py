# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (dml)
#     language: python
#     name: dml
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.11.14
# ---

# %% [markdown]
# # 00. Environment Setup and Verification
#
# This notebook verifies the Python environment, checks required packages,
# and sets up the project paths for reproducibility.
#
# **Purpose:**
# - Verify all required packages are installed
# - Set up project paths
# - Create spec ledger for tracking specifications
# - Define global constants and configuration

# %%
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Project root (relative to notebook location)
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in dir() else Path.cwd().parent
os.chdir(PROJECT_ROOT)

print(f"Python version: {sys.version}")
print(f"Project root: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Package Verification

# %%
REQUIRED_PACKAGES = [
    'pandas',
    'numpy',
    'openpyxl',
    'requests',
    'pycountry',
    'matplotlib',
    'seaborn',
    'statsmodels',
    'scipy',
    'jupytext',
    'tqdm',
]

missing_packages = []
installed_versions = {}

for pkg in REQUIRED_PACKAGES:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        installed_versions[pkg] = version
        print(f"✓ {pkg}: {version}")
    except ImportError:
        missing_packages.append(pkg)
        print(f"✗ {pkg}: NOT INSTALLED")

if missing_packages:
    print(f"\n⚠ Missing packages: {missing_packages}")
    print("Install with: pip install " + " ".join(missing_packages))
else:
    print("\n✓ All required packages installed!")

# %% [markdown]
# ## 2. Directory Structure Verification

# %%
REQUIRED_DIRS = [
    'data/01_raw',
    'data/02_intermediate',
    'data/03_clean',
    'output/figures',
    'output/tables',
    'output/logs',
    'paper/figures',
    'paper/tables',
    'paper/notes',
    'notebooks',
]

print("Directory structure verification:")
for dir_path in REQUIRED_DIRS:
    full_path = PROJECT_ROOT / dir_path
    if full_path.exists():
        print(f"✓ {dir_path}")
    else:
        print(f"✗ {dir_path} - CREATING...")
        full_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 3. Fraser Data Verification

# %%
fraser_path = PROJECT_ROOT / 'data/01_raw/fraser.xlsx'
assert fraser_path.exists(), f"Fraser data not found at {fraser_path}"
print(f"✓ Fraser data found: {fraser_path}")
print(f"  File size: {fraser_path.stat().st_size / 1e6:.2f} MB")

# %% [markdown]
# ## 4. Global Configuration

# %%
# Quinquennial configuration
CONFIG = {
    # Quinquennial grid
    'quinquennial_years': list(range(1970, 2025, 5)),  # 1970, 1975, ..., 2020
    
    # Shock thresholds (main and robustness)
    'shock_thresholds': {
        'main': 1.0,  # Primary threshold: 1 point change
        'robustness': [0.75, 1.25, 1.5],  # Robustness thresholds
    },
    
    # Sustained reform parameters
    'sustained_reform': {
        'epsilon': 0.5,  # Tolerance for decline
        'periods_forward': 2,  # Check t+1 and t+2
    },
    
    # Cooldown period (number of quinquennial periods)
    'cooldown_periods': 2,
    
    # Event study horizons
    'horizons': {
        'pre_periods': 2,  # K in the paper (-2, -1)
        'post_periods': 4,  # L in the paper (0, 1, 2, 3, 4)
    },
    
    # World Bank indicators
    'worldbank_indicators': {
        'gdp_pc_constant': 'NY.GDP.PCAP.KD',
        'gdp_pc_ppp': 'NY.GDP.PCAP.PP.KD',
        'population': 'SP.POP.TOTL',
    },
    
    # Date of run
    'run_date': datetime.now().isoformat(),
    
    # Random seed for reproducibility
    'random_seed': 42,
}

print("Configuration:")
for key, value in CONFIG.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {key}: {value}")

# %% [markdown]
# ## 5. Initialize Specification Ledger

# %%
spec_ledger_path = PROJECT_ROOT / 'output/logs/spec_ledger.json'

# Initialize or load existing ledger
if spec_ledger_path.exists():
    with open(spec_ledger_path, 'r') as f:
        spec_ledger = json.load(f)
    print(f"✓ Loaded existing spec ledger with {len(spec_ledger.get('runs', []))} previous runs")
else:
    spec_ledger = {
        'created': datetime.now().isoformat(),
        'runs': [],
        'specifications': {}
    }
    print("✓ Created new spec ledger")

# Add current configuration as a run
current_run = {
    'timestamp': datetime.now().isoformat(),
    'config': CONFIG,
    'python_version': sys.version,
    'packages': installed_versions,
}
spec_ledger['runs'].append(current_run)

# Save ledger
with open(spec_ledger_path, 'w') as f:
    json.dump(spec_ledger, f, indent=2, default=str)

print(f"✓ Spec ledger saved to {spec_ledger_path}")

# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "="*60)
print("ENVIRONMENT SETUP COMPLETE")
print("="*60)
print(f"Python: {sys.version.split()[0]}")
print(f"Project: {PROJECT_ROOT.name}")
print(f"Quinquennial years: {CONFIG['quinquennial_years'][0]}-{CONFIG['quinquennial_years'][-1]}")
print(f"Main shock threshold: {CONFIG['shock_thresholds']['main']}")
print(f"Event horizons: K={CONFIG['horizons']['pre_periods']}, L={CONFIG['horizons']['post_periods']}")
print("="*60)
