import json
import time
import sys
import nbformat
import traceback
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

# Reconfigure stdout for UTF-8
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = "notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb"

print(f"Loading and repairing notebook: {notebook_path}")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb_dict = json.load(f)

# Fix corrupted metadata and source lists
repaired_metadata = 0
repaired_source = 0
for idx, cell in enumerate(nb_dict['cells']):
    # Fix metadata
    if not isinstance(cell.get('metadata'), dict):
        cell['metadata'] = {}
        repaired_metadata += 1
    # Fix source to string if it is a list
    source = cell.get('source', '')
    if isinstance(source, list):
        cell['source'] = "".join(source)
        repaired_source += 1

print(f"Repaired {repaired_metadata} cells with invalid metadata types.")
print(f"Formatted {repaired_source} cell sources from lists to strings.")

try:
    # Convert to nbformat version 4 object
    nb = nbformat.from_dict(nb_dict)
    
    print("Starting execution of notebook cells...")
    ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
    
    start_time = time.time()
    ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
    end_time = time.time()
    print(f"Notebook executed successfully in {end_time - start_time:.2f} seconds!")
    
    # Save the executed notebook back
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Executed notebook saved successfully with cell outputs.")
except CellExecutionError as e:
    print(f"\nCRITICAL: Cell execution failed!")
    print(f"Error cell index: {e.cell_index if hasattr(e, 'cell_index') else 'Unknown'}")
    print(f"Error details:\n{e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()
    sys.exit(1)
