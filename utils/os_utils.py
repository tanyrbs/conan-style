import logging
import os
import shutil
from pathlib import Path


def link_file(from_file, to_file):
    from_path = Path(from_file)
    to_path = Path(to_file)
    try:
        if to_path.exists() or to_path.is_symlink():
            to_path.unlink()
    except Exception:
        pass
    try:
        relative_target = os.path.relpath(str(from_path), start=str(to_path.parent))
        os.symlink(relative_target, str(to_path))
    except Exception:
        # Fallback to copy when symlink is unavailable (e.g., Windows without privileges).
        copy_file(from_file, to_file)


def move_file(from_file, to_file):
    shutil.move(from_file, to_file)


def copy_file(from_file, to_file):
    from_path = Path(from_file)
    to_path = Path(to_file)
    if from_path.is_dir():
        if to_path.exists():
            shutil.rmtree(to_path, ignore_errors=True)
        shutil.copytree(from_path, to_path)
    else:
        to_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(from_path, to_path)


def remove_file(*fns):
    for f in fns:
        try:
            path = Path(f)
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=False)
            else:
                path.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:
            logging.warning("Failed to remove %s: %s", f, exc)
