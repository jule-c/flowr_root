import os
import subprocess

from posecheck.utils.constants import REDUCE_PATH

def is_reduce_installed(reduce_path: str = REDUCE_PATH) -> bool:
    """
    Check if the 'reduce' executable at the given path is installed and working.

    Args:
        reduce_path: The path to the 'reduce' executable.

    Returns:
        True if 'reduce' is installed and working, False otherwise.
    """
    try:
        result = subprocess.run([reduce_path, '-version'], capture_output=True, text=True)
        return result.returncode == 1
    except Exception as e:
        print(f"An error occurred while checking 'reduce': {e}")
        return False
    
def print_reduce_warning():
    print("***** WARNING: reduce is not installed      *****")
    print("***** WARNING: clashes and interaction fingerprinting may not work *****")
    print("***** WARNING: we highly recommend using reduce as in the paper for comparison *****")
    print("***** WARNING: Install instructions in README.md *****")

if __name__ == "__main__":
    print(is_reduce_installed())