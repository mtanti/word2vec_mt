@echo off

call conda activate venv\ || pause && exit /b

echo #########################################
echo mypy
for /f %%F in ('dir *.py /b') do (
    if %%F neq tokenise.py (
        echo ..checking %%F
        call python -m mypy %%F --follow-imports=silent || pause && exit /b
    )
)

echo #########################################
echo pylint
for /f %%F in ('dir *.py /b') do (
    if %%F neq tokenise.py (
        echo ..checking %%F
        call python -m pylint %%F || pause && exit /b
    )
)
