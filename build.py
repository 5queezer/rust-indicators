# File: rust_indicators/build.py
"""
Build script for Rust indicators
Usage: python build.py
"""

import subprocess
import sys
from pathlib import Path

def build_rust_module():
    """Build the Rust module using maturin"""
    rust_dir = Path(__file__).parent

    print("🦀 Building Rust indicators module...")

    try:
        # Install maturin if not available
        subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)

        # Build and install the module
        result = subprocess.run([
            "maturin", "develop", "--release"
        ], cwd=rust_dir, check=True, capture_output=True, text=True)

        print("✅ Rust module built successfully!")
        print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Maturin not found. Install with: pip install maturin")
        return False

if __name__ == "__main__":
    success = build_rust_module()
    sys.exit(0 if success else 1)

