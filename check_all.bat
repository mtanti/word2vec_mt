@echo off

call conda activate venv\ || pause && exit /b

echo #########################################
echo mypy
for /f %%F in ('dir *.py /b') do (
    echo ..checking %%F
    call python -m mypy %%F || pause && exit /b
)

echo #########################################
echo pylint
for /f %%F in ('dir *.py /b') do (
    echo ..checking %%F
    call python -m pylint %%F || pause && exit /b
)
