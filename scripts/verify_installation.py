#!/usr/bin/env python3
"""
Verify OKADFA installation and dependencies
"""

import sys


def check_imports():
    """Check if all required packages can be imported"""
    packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "einops": "Einops",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "scipy": "SciPy",
        "yaml": "PyYAML",
        "omegaconf": "OmegaConf",
    }
    
    print("Checking package imports...")
    all_ok = True
    
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {name:20s} (version: {version})")
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED: {e}")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Check CUDA availability"""
    import torch
    
    print("\nChecking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("✗ CUDA is not available (CPU-only mode)")
        return False


def check_project_structure():
    """Verify project structure"""
    import os
    from pathlib import Path
    
    print("\nChecking project structure...")
    base_path = Path(__file__).parent.parent
    
    required_dirs = [
        "src",
        "src/models",
        "src/training",
        "src/kernels",
        "src/diagnostics",
        "configs",
        "tests",
        "scripts",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_ok = False
    
    return all_ok


def main():
    print("=" * 60)
    print("OKADFA Installation Verification")
    print("=" * 60)
    print()
    
    imports_ok = check_imports()
    cuda_ok = check_cuda()
    structure_ok = check_project_structure()
    
    print("\n" + "=" * 60)
    if imports_ok and structure_ok:
        print("✓ Installation verification PASSED")
        print("\nYou can now start implementing OKADFA components!")
        return 0
    else:
        print("✗ Installation verification FAILED")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
