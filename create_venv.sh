#!/bin/bash
set -e

conda create --yes --prefix venv/ python=3.9
conda shell.bash activate venv/

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e .
