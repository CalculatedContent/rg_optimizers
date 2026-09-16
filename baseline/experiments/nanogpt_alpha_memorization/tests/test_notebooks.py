"""Notebook templates stay parseable by the repository's Python AST checker."""
import ast
import json
from pathlib import Path


def test_every_notebook_code_cell_parses():
    paths=list((Path(__file__).resolve().parents[1]/'notebooks').glob('*.ipynb'))
    assert len(paths)==3
    for path in paths:
        for cell in json.loads(path.read_text())['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']),filename=str(path))
