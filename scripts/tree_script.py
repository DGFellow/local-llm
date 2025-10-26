import os

EXCLUDE_DIRS = {'node_modules','venv','.venv', '__pycache__', '.git', 'docs', 'corrections','.pytest_cache', 'tools'}

def tree(dir_path, indent="", output_lines=None):
    if output_lines is None:
        output_lines = []

    items = sorted(os.listdir(dir_path))
    for i, item in enumerate(items):
        if item in EXCLUDE_DIRS:
            continue
        full_path = os.path.join(dir_path, item)
        is_last = (i == len(items) - 1)
        prefix = "└── " if is_last else "├── "
        output_lines.append(indent + prefix + item)
        if os.path.isdir(full_path):
            new_indent = indent + ("    " if is_last else "│   ")
            tree(full_path, new_indent, output_lines)

    return output_lines

if __name__ == "__main__":
    root_dir = "."  # Change to your path if needed
    lines = tree(root_dir)
    with open("tree.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
