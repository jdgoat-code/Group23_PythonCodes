@echo off
rem STEP 2 (Windows): opens the Streamlit interface in your browser.
rem Double-click this file, or run it from Command Prompt.
rem Stop the app any time with Ctrl+C.

cd /d "%~dp0" || goto :directory_error

if not exist ".venv\Scripts\python.exe" (
    echo Setup has not been run yet. In this folder, run:
    echo   py -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo   .venv\Scripts\python.exe train_model.py
    goto :end
)

if not exist "complaint_pipeline.joblib" (
    echo No trained model found. Run .venv\Scripts\python.exe train_model.py first.
    goto :end
)

echo Starting the app... your browser will open at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py
goto :end

:directory_error
echo Could not open the app folder.

:end
echo.
pause
