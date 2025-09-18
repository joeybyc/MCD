import os
import shutil

def remove_pycache(root_dir='.'):
    for root, dirs, files in os.walk(root_dir):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            shutil.rmtree(cache_path)
            print(f"Removed: {cache_path}")

if __name__ == "__main__":
    remove_pycache()