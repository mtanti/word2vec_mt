#!/bin/bash
set -e

conda activate venv/

echo "#########################################"
for FNAME in `find -maxdepth 1 -name "*.py"`
do
    echo "..checking $FNAME"
    python -m mypy $FNAME
done
echo ""

echo "#########################################"
echo "pylint"
for FNAME in `find -maxdepth 1 -name "*.py"`
do
    echo "..checking $FNAME"
    python -m pylint $FNAME
done
