@echo off
rem STEP 1 (Windows): installs everything and trains the model. Run this once.
rem Double-click this file, or run it from Command Prompt.

cd /d "%~dp0" || goto :directory_error

if not exist "test_records.csv" (
    echo test_records.csv is missing. Put it in the same folder as this file.
    goto :end
)

py -3 -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py -3"
) else (
    python -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1
    if errorlevel 1 (
        echo Python 3.9 or newer was not found.
        echo Install it from https://www.python.org/downloads/windows/ and enable Add Python to PATH.
        goto :end
    )
    set "PYTHON=python"
)

echo ==^> Creating a private Python environment (.venv)...
%PYTHON% -m venv .venv
if errorlevel 1 (
    echo Could not create the environment.
    goto :end
)

echo ==^> Installing packages (this can take a few minutes the first time)...
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Install failed. Check your internet connection.
    goto :end
)

echo ==^> Training the models...
".venv\Scripts\python.exe" train_model.py
if errorlevel 1 (
    echo Training failed. Copy the error above and ask for help.
    goto :end
)

echo.
echo DONE. Created in this folder:
echo   complaint_pipeline.joblib, model_comparison.csv/.png, confusion_*.png
echo Next: double-click run_app.bat
goto :end

:directory_error
echo Could not open the app folder.

:end
echo.
pause
