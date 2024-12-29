import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python():
    """Check Python version."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        sys.exit(1)

def setup_virtual_env():
    """Create and activate virtual environment."""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Get the path to the virtual environment Python
    if os.name == "nt":  # Windows
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Unix
        python_path = venv_path / "bin" / "python"
    
    return str(python_path)

def install_dependencies(python_path):
    """Install required packages."""
    print("Installing dependencies...")
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([python_path, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    subprocess.run([python_path, "-m", "pip", "install", "-e", "."], check=True)

def setup_env_file():
    """Create .env file if it doesn't exist."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists() and env_example_path.exists():
        print("Creating .env file from template...")
        shutil.copy(env_example_path, env_path)
        print("Please edit .env file with your API keys")

def main():
    """Run installation process."""
    print("=== Installing InCA System ===\n")
    
    try:
        # Check Python version
        check_python()
        
        # Setup virtual environment
        python_path = setup_virtual_env()
        
        # Install dependencies
        install_dependencies(python_path)
        
        # Setup environment file
        setup_env_file()
        
        print("\n=== Installation Complete ===")
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if os.name == "nt":  # Windows
            print("   .\\venv\\Scripts\\activate")
        else:  # Unix
            print("   source venv/bin/activate")
        print("2. Edit the .env file with your API keys")
        print("3. Run verify_installation.py to check the setup:")
        print("   python verify_installation.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
