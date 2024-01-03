#!/bin/bash
set -e

conda create --prefix venv/ python=3.9
conda activate venv/

pip install -r requirements.txt

bash check_all.sh
