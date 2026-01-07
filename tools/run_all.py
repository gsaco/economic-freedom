#!/usr/bin/env python3
"""
run_all.py — Master Pipeline Executor for Economic Freedom Analysis

This script:
1. Discovers and runs all notebooks/*.py in numeric order
2. Captures stdout/stderr, runtime, exit codes
3. Tracks artifact creation/modification
4. Writes run ledger and per-notebook logs
5. Calls build_report.py to generate the PhD-level report

Usage:
    python tools/run_all.py
"""

import os
import sys
import json
import subprocess
import time
import re
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
TOOLS_DIR = PROJECT_ROOT / "tools"

# Python interpreter to use
PYTHON_EXECUTABLE = sys.executable

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dirs():
    """Ensure output directories exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)


def get_notebook_order(notebooks_dir: Path) -> List[Path]:
    """Get notebooks sorted by leading numeric prefix."""
    py_files = list(notebooks_dir.glob("*.py"))
    
    def extract_number(p: Path) -> int:
        match = re.match(r'^(\d+)', p.stem)
        return int(match.group(1)) if match else 999
    
    return sorted(py_files, key=extract_number)


def snapshot_output_files(output_dir: Path) -> Dict[str, float]:
    """Snapshot all files under output/ with their mtimes."""
    snapshot = {}
    for root, dirs, files in os.walk(output_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            fp = Path(root) / f
            try:
                snapshot[str(fp.relative_to(output_dir))] = fp.stat().st_mtime
            except:
                pass
    return snapshot


def diff_snapshots(before: Dict, after: Dict) -> Dict[str, List[str]]:
    """Find created and modified files between snapshots."""
    created = []
    modified = []
    
    for path, mtime in after.items():
        if path not in before:
            created.append(path)
        elif before[path] != mtime:
            modified.append(path)
    
    return {"created": created, "modified": modified}


def run_notebook(notebook_path: Path, log_dir: Path) -> Dict[str, Any]:
    """
    Run a single notebook and capture all outputs.
    
    Returns dict with:
        - notebook: name
        - status: 'success' or 'failed'
        - exit_code: int
        - runtime_seconds: float
        - stdout_tail: last 40 lines
        - stderr_tail: last 40 lines
        - log_file: path to full log
        - warnings_count: int (estimated)
    """
    notebook_name = notebook_path.stem
    log_file = log_dir / f"run_{notebook_name}.log"
    
    result = {
        "notebook": notebook_name,
        "notebook_file": notebook_path.name,
        "status": "unknown",
        "exit_code": None,
        "runtime_seconds": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "log_file": str(log_file.relative_to(PROJECT_ROOT)),
        "warnings_count": 0,
        "start_time": None,
        "end_time": None,
    }
    
    start_time = time.time()
    result["start_time"] = datetime.now().isoformat()
    
    try:
        proc = subprocess.run(
            [PYTHON_EXECUTABLE, str(notebook_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout per notebook
        )
        
        end_time = time.time()
        result["end_time"] = datetime.now().isoformat()
        result["runtime_seconds"] = round(end_time - start_time, 2)
        result["exit_code"] = proc.returncode
        result["status"] = "success" if proc.returncode == 0 else "failed"
        
        # Store full output
        full_output = f"=== STDOUT ===\n{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}"
        with open(log_file, "w") as f:
            f.write(full_output)
        
        # Extract tails
        stdout_lines = proc.stdout.strip().split("\n")
        stderr_lines = proc.stderr.strip().split("\n")
        result["stdout_tail"] = "\n".join(stdout_lines[-40:])
        result["stderr_tail"] = "\n".join(stderr_lines[-40:]) if stderr_lines != [''] else ""
        
        # Count warnings
        combined = proc.stdout + proc.stderr
        result["warnings_count"] = combined.lower().count("warning")
        
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["exit_code"] = -1
        result["runtime_seconds"] = 600
        with open(log_file, "w") as f:
            f.write("TIMEOUT: Notebook exceeded 10 minute limit")
    except Exception as e:
        result["status"] = "error"
        result["exit_code"] = -2
        result["runtime_seconds"] = time.time() - start_time
        with open(log_file, "w") as f:
            f.write(f"EXCEPTION: {str(e)}")
    
    return result


def get_git_info() -> Dict[str, str]:
    """Get git commit info if available."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        return {
            "commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
            "branch": branch.stdout.strip() if branch.returncode == 0 else "unknown"
        }
    except:
        return {"commit": "unknown", "branch": "unknown"}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("ECONOMIC FREEDOM ANALYSIS — PIPELINE EXECUTOR")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    
    # Setup
    ensure_dirs()
    
    # Get notebooks
    notebooks = get_notebook_order(NOTEBOOKS_DIR)
    print(f"Found {len(notebooks)} notebooks:")
    for nb in notebooks:
        print(f"  - {nb.name}")
    print()
    
    # Run ledger
    run_ledger = {
        "run_timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "git": get_git_info(),
        "notebooks": [],
        "summary": {
            "total": len(notebooks),
            "success": 0,
            "failed": 0,
            "total_runtime_seconds": 0
        }
    }
    
    all_success = True
    
    # Run each notebook
    for i, notebook in enumerate(notebooks):
        print("-" * 70)
        print(f"[{i+1}/{len(notebooks)}] Running: {notebook.name}")
        print("-" * 70)
        
        # Snapshot before
        snapshot_before = snapshot_output_files(OUTPUT_DIR)
        
        # Run notebook
        result = run_notebook(notebook, LOGS_DIR)
        
        # Snapshot after
        snapshot_after = snapshot_output_files(OUTPUT_DIR)
        artifact_diff = diff_snapshots(snapshot_before, snapshot_after)
        
        # Store artifact info
        artifact_info = {
            "notebook": result["notebook"],
            "created": artifact_diff["created"],
            "modified": artifact_diff["modified"],
            "timestamp": datetime.now().isoformat()
        }
        
        artifact_file = LOGS_DIR / f"artifacts_{result['notebook']}.json"
        with open(artifact_file, "w") as f:
            json.dump(artifact_info, f, indent=2)
        
        result["artifacts"] = artifact_info
        run_ledger["notebooks"].append(result)
        
        # Update summary
        if result["status"] == "success":
            run_ledger["summary"]["success"] += 1
            status_icon = "✓"
        else:
            run_ledger["summary"]["failed"] += 1
            all_success = False
            status_icon = "✗"
        
        if result["runtime_seconds"]:
            run_ledger["summary"]["total_runtime_seconds"] += result["runtime_seconds"]
        
        # Print result
        print(f"{status_icon} Status: {result['status']} (exit code: {result['exit_code']})")
        print(f"  Runtime: {result['runtime_seconds']}s")
        print(f"  Artifacts created: {len(artifact_diff['created'])}")
        print(f"  Artifacts modified: {len(artifact_diff['modified'])}")
        if artifact_diff['created']:
            for a in artifact_diff['created'][:5]:
                print(f"    + {a}")
        print()
    
    # Save run ledger
    ledger_path = LOGS_DIR / "run_ledger.json"
    with open(ledger_path, "w") as f:
        json.dump(run_ledger, f, indent=2)
    print(f"✓ Saved run ledger to: {ledger_path}")
    
    # Build report
    print()
    print("=" * 70)
    print("BUILDING REPORT")
    print("=" * 70)
    
    report_script = TOOLS_DIR / "build_report.py"
    if report_script.exists():
        result = subprocess.run(
            [PYTHON_EXECUTABLE, str(report_script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    else:
        print(f"Warning: {report_script} not found, skipping report generation")
    
    # Final summary
    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total notebooks: {run_ledger['summary']['total']}")
    print(f"Successful: {run_ledger['summary']['success']}")
    print(f"Failed: {run_ledger['summary']['failed']}")
    print(f"Total runtime: {run_ledger['summary']['total_runtime_seconds']:.1f}s")
    print()
    print(f"Run ledger: {ledger_path}")
    print(f"Report: {OUTPUT_DIR / 'report.md'}")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
