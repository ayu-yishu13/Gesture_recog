import os

EXCLUDE = {"mp_env", "venv", "__pycache__", ".git"}

def print_tree(path, prefix=""):
    for name in sorted(os.listdir(path)):
        if name in EXCLUDE:
            continue
        full_path = os.path.join(path, name)
        print(prefix + "├── " + name)
        if os.path.isdir(full_path):
            print_tree(full_path, prefix + "│   ")

print_tree(".")
