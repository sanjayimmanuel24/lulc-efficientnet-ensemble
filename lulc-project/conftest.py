"""
Root-level conftest.py.

Pytest loads this before any test collection begins. Adding the project
root to sys.path here means `import src.models.fusion` works regardless
of whether the user runs `pytest` from inside the lulc-project/ folder,
from a parent folder, or from an IDE like PyCharm/VS Code that sets its
own working directory.

This is the standard, cross-platform fix for the "ModuleNotFoundError:
No module named 'src'" error on Windows.
"""
import sys
from pathlib import Path

# Insert the directory that *contains* the src/ package (i.e. lulc-project/)
# at the front of sys.path so it takes priority over any installed packages
# with the same name.
sys.path.insert(0, str(Path(__file__).parent))
