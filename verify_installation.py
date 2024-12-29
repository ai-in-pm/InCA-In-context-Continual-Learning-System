import os
import sys
import importlib
from importlib import metadata
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

@dataclass
class VerificationResult:
    """Store verification results for reporting."""
    status: bool
    message: str
    details: Optional[Dict] = None

class InstallationVerifier:
    """Verify InCA installation and dependencies."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("logs/installation.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[VerificationResult] = []
    
    def log_result(self, result: VerificationResult):
        """Log verification result to file."""
        with self.log_file.open("a") as f:
            f.write(f"[{'PASS' if result.status else 'FAIL'}] {result.message}\n")
            if result.details:
                for key, value in result.details.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    def check_python_version(self) -> VerificationResult:
        """Check if Python version is compatible."""
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version < required_version:
            return VerificationResult(
                status=False,
                message=f"Python version {current_version[0]}.{current_version[1]} is not supported. "
                        f"Please use Python {required_version[0]}.{required_version[1]} or higher.",
                details={"current_version": f"{current_version[0]}.{current_version[1]}",
                        "required_version": f"{required_version[0]}.{required_version[1]}"}
            )
        
        return VerificationResult(
            status=True,
            message=f"Python version {current_version[0]}.{current_version[1]} is compatible",
            details={"python_version": f"{current_version[0]}.{current_version[1]}"}
        )
    
    def check_dependencies(self) -> List[VerificationResult]:
        """Check if all required packages are installed."""
        required_packages = {
            "numpy": "1.24.0",
            "scipy": "1.10.0",
            "openai": "1.3.0",
            "anthropic": "0.3.0",
            "google-generativeai": "0.3.0",
            "mistralai": "0.0.7",
            "transformers": "4.35.0",
            "torch": "2.1.0",
            "pandas": "2.1.0",
            "matplotlib": "3.7.0",
            "seaborn": "0.12.0",
            "pytest": "7.4.0",
            "jupyter": "1.0.0",
            "scikit-learn": "1.3.0",
            "python-dotenv": "1.0.0"
        }
        
        results = []
        for package, min_version in required_packages.items():
            try:
                installed_version = metadata.version(package)
                if metadata.version(package) < min_version:
                    results.append(VerificationResult(
                        status=False,
                        message=f"{package} version {installed_version} is below minimum required version {min_version}",
                        details={"package": package,
                                "installed_version": installed_version,
                                "required_version": min_version}
                    ))
                else:
                    results.append(VerificationResult(
                        status=True,
                        message=f"{package} version {installed_version} is compatible",
                        details={"package": package, "version": installed_version}
                    ))
            except metadata.PackageNotFoundError:
                results.append(VerificationResult(
                    status=False,
                    message=f"{package} is not installed",
                    details={"package": package, "required_version": min_version}
                ))
        
        return results
    
    def check_api_keys(self) -> List[VerificationResult]:
        """Check if all required API keys are set."""
        load_dotenv()
        required_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "MISTRAL_API_KEY",
            "GROQ_API_KEY"
        ]
        
        results = []
        for key in required_keys:
            if not os.getenv(key):
                results.append(VerificationResult(
                    status=False,
                    message=f"{key} is not set in .env file",
                    details={"key": key, "status": "missing"}
                ))
            else:
                results.append(VerificationResult(
                    status=True,
                    message=f"{key} is set",
                    details={"key": key, "status": "present"}
                ))
        
        return results
    
    def check_imports(self) -> List[VerificationResult]:
        """Check if all package imports work correctly."""
        required_imports = [
            ("numpy", "np"),
            ("scipy.stats", "stats"),
            ("openai", None),
            ("anthropic", None),
            ("google.generativeai", "genai"),
            ("mistralai.client", "MistralClient"),
            ("transformers", None),
            ("torch", None),
            ("pandas", "pd"),
            ("matplotlib.pyplot", "plt"),
            ("seaborn", "sns"),
            ("pytest", None),
            ("jupyter", None),
            ("sklearn", None)
        ]
        
        results = []
        for module, alias in required_imports:
            try:
                importlib.import_module(module)
                results.append(VerificationResult(
                    status=True,
                    message=f"Successfully imported {module}",
                    details={"module": module, "alias": alias}
                ))
            except ImportError as e:
                results.append(VerificationResult(
                    status=False,
                    message=f"Failed to import {module}: {str(e)}",
                    details={"module": module, "error": str(e)}
                ))
        
        return results
    
    def check_system_requirements(self) -> List[VerificationResult]:
        """Check system requirements like disk space and memory."""
        results = []
        
        # Check available disk space
        try:
            total, used, free = os.statvfs(".").f_blocks, os.statvfs(".").f_bfree, os.statvfs(".").f_bavail
            free_gb = (free * os.statvfs(".").f_frsize) / (1024**3)
            if free_gb < 1.0:  # Less than 1GB free
                results.append(VerificationResult(
                    status=False,
                    message=f"Low disk space: {free_gb:.2f}GB available",
                    details={"free_space_gb": f"{free_gb:.2f}"}
                ))
            else:
                results.append(VerificationResult(
                    status=True,
                    message=f"Sufficient disk space: {free_gb:.2f}GB available",
                    details={"free_space_gb": f"{free_gb:.2f}"}
                ))
        except Exception as e:
            results.append(VerificationResult(
                status=True,  # Don't fail on check error
                message="Could not check disk space",
                details={"error": str(e)}
            ))
        
        # Add more system checks here
        return results
    
    def verify_installation(self, verbose: bool = False) -> bool:
        """Run all verification checks."""
        print("\n=== Checking InCA Installation ===\n")
        
        all_results = []
        
        # Check Python version
        print("\n--- Checking Python Version ---")
        result = self.check_python_version()
        self.log_result(result)
        print(f"[{'PASS' if result.status else 'FAIL'}] {result.message}")
        all_results.append(result)
        
        # Check dependencies
        print("\n--- Checking Dependencies ---")
        results = self.check_dependencies()
        for result in results:
            self.log_result(result)
            print(f"[{'PASS' if result.status else 'FAIL'}] {result.message}")
            all_results.append(result)
        
        # Check API keys
        print("\n--- Checking API Keys ---")
        results = self.check_api_keys()
        for result in results:
            self.log_result(result)
            print(f"[{'PASS' if result.status else 'FAIL'}] {result.message}")
            all_results.append(result)
        
        # Check imports
        print("\n--- Checking Imports ---")
        results = self.check_imports()
        for result in results:
            self.log_result(result)
            print(f"[{'PASS' if result.status else 'FAIL'}] {result.message}")
            all_results.append(result)
        
        # Check system requirements
        print("\n--- Checking System Requirements ---")
        results = self.check_system_requirements()
        for result in results:
            self.log_result(result)
            print(f"[{'PASS' if result.status else 'FAIL'}] {result.message}")
            all_results.append(result)
        
        # Print summary
        print("\n=== Installation Verification Summary ===")
        total_checks = len(all_results)
        passed_checks = sum(1 for r in all_results if r.status)
        
        if verbose:
            print("\nDetailed Results:")
            for result in all_results:
                print(f"\n[{'PASS' if result.status else 'FAIL'}] {result.message}")
                if result.details:
                    for key, value in result.details.items():
                        print(f"  {key}: {value}")
        
        success = all(r.status for r in all_results)
        if success:
            print(f"\n[PASS] All checks passed! ({passed_checks}/{total_checks})")
            print(f"InCA is ready to use.")
        else:
            print(f"\n[FAIL] Some checks failed. ({passed_checks}/{total_checks} passed)")
            print(f"Please check logs/installation.log for details.")
        
        return success

def main():
    """Run installation verification."""
    import argparse
    parser = argparse.ArgumentParser(description="Verify InCA installation")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results")
    args = parser.parse_args()
    
    verifier = InstallationVerifier()
    success = verifier.verify_installation(verbose=args.verbose)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
