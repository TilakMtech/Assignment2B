"""Exports the saved notebook to HTML (close the notebook tab first)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_notebook import export  # noqa: E402

if __name__ == "__main__":
    export()
    print("Exported.")
