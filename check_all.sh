#!/bin/bash
set -e

conda activate venv/

echo "#########################################"
for FNAME in `find -maxdepth 1 -name "*.py"`
do
    if [$FNAME -ne 'tokenise.py']; then
        echo "..checking $FNAME"
        python -m mypy $FNAME --follow-imports=silent
    fi
done
echo ""

echo "#########################################"
echo "pylint"
for FNAME in `find -maxdepth 1 -name "*.py"`
do
    if [$FNAME -ne 'tokenise.py']; then
        echo "..checking $FNAME"
        python -m pylint $FNAME
    fi
done
