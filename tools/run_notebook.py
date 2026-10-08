"""Runs the whole notebook headless in the python_llm kernel, saving after every cell,
then exports the HTML. Use instead of Run All (no autosave / stale-HTML problems).

    KERNEL=python_llm python -u tools/run_notebook.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "Group4_Assignment2B_RAG.ipynb"
KERNEL = os.environ.get("KERNEL", "python_llm")


def export():
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    extra = [str(Path(p) / "share" / "jupyter") for p in {sys.base_prefix, "/opt/conda"}]
    env["JUPYTER_PATH"] = os.pathsep.join([env.get("JUPYTER_PATH", "")] + extra).strip(os.pathsep)
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(NB), "--output-dir", str(NB.parent)],
                   check=True, env=env)


def main():
    nb = nbformat.read(NB, as_version=4)
    os.environ["PYTHONNOUSERSITE"] = "1"
    client = NotebookClient(nb, kernel_name=KERNEL, timeout=7200, resources={"metadata": {"path": str(NB.parent)}})
    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "nbconvert" not in c.source]
    with client.setup_kernel():
        for n, i in enumerate(code_cells, 1):
            first = nb.cells[i].source.splitlines()[0][:70]
            print(f"[{n}/{len(code_cells)}] {first}", flush=True)
            t0 = time.time()
            try:
                client.execute_cell(nb.cells[i], i)
            except CellExecutionError as err:
                nbformat.write(nb, NB)
                sys.exit(f"Cell {i} failed (outputs so far are saved in the notebook):\n{err}")
            nbformat.write(nb, NB)
            print(f"    done in {time.time() - t0:.0f}s", flush=True)
    export()
    print("Notebook executed and exported:", NB.with_suffix(".html"))


if __name__ == "__main__":
    main()
