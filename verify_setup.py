#!/usr/bin/env python3
"""
Verification script for Career Intelligence Assistant development environment.
Run after activating venv and installing dependencies to confirm Step 1 setup is complete.
"""

import os
import sys
from pathlib import Path

# Project root (where this script lives)
PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_ENV_KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
OPTIONAL_ENV_KEYS = ["VECTOR_DB_PATH"]


def check_folder(path: Path, description: str) -> bool:
    """Check if folder exists; create if missing for data dirs."""
    exists = path.exists()
    if not exists and "data" in str(path):
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {path.relative_to(PROJECT_ROOT)}")
            return True
        except OSError as e:
            print(f"  Failed to create {path}: {e}")
            return False
    return exists


def check_init_file(module_path: Path) -> bool:
    """Check if __init__.py exists in module directory."""
    init_file = module_path / "__init__.py"
    return init_file.exists()


def validate_env_keys() -> bool:
    """Check .env has required keys (without exposing values)."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("  .env file not found")
        return False

    # Parse .env: look for KEY=value (ignore comments, empty lines)
    found_keys = set()
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key = line.split("=")[0].strip()
                value = line.split("=", 1)[1].strip()
                if value and not value.startswith("#"):
                    found_keys.add(key)

    missing = [k for k in REQUIRED_ENV_KEYS if k not in found_keys]
    if missing:
        print(f"  Missing required keys in .env: {', '.join(missing)}")
        return False

    # Check values are non-placeholder (optional strict check)
    return True


def main() -> int:
    """Run all verification checks."""
    print("Career Intelligence Assistant - Setup Verification\n")
    all_ok = True

    # 1. Required folders
    print("1. Checking folders...")
    folders = [
        (PROJECT_ROOT / "data", "Data root"),
        (PROJECT_ROOT / "data" / "uploads", "Uploads"),
        (PROJECT_ROOT / "data" / "vectordb", "Vector DB"),
        (PROJECT_ROOT / "tests" / "fixtures", "Test fixtures"),
    ]
    for path, desc in folders:
        if check_folder(path, desc):
            print(f"  OK: {path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  MISSING: {path.relative_to(PROJECT_ROOT)}")
            all_ok = False
    print()

    # 2. __init__.py files
    print("2. Checking __init__.py files...")
    modules = [
        "app/ingestion",
        "app/retrieval",
        "app/rag",
        "app/api",
        "app/utils",
        "tests",
    ]
    for mod in modules:
        path = PROJECT_ROOT / mod
        if check_init_file(path):
            print(f"  OK: {mod}/__init__.py")
        else:
            print(f"  MISSING: {mod}/__init__.py")
            all_ok = False
    print()

    # 3. .env validation
    print("3. Validating .env keys...")
    if validate_env_keys():
        print("  OK: Required keys present (ANTHROPIC_API_KEY, OPENAI_API_KEY)")
    else:
        all_ok = False
    print()

    # 4. Key files
    print("4. Checking key files...")
    key_files = [
        "app/main.py",
        "app/config.py",
        "app/models.py",
        "requirements.txt",
        ".env.example",
    ]
    for f in key_files:
        if (PROJECT_ROOT / f).exists():
            print(f"  OK: {f}")
        else:
            print(f"  MISSING: {f}")
            all_ok = False
    print()

    if all_ok:
        print("=" * 50)
        print("  Setup Complete")
        print("=" * 50)
        print("\nNext: Run 'uvicorn app.main:app --reload' to start the API.")
        return 0
    else:
        print("=" * 50)
        print("  Setup incomplete - fix the issues above")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
