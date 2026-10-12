@echo off
rem Retrain the model using the current test_records.csv and existing .venv.

cd /d "%~dp0" || goto :directory_error

if not exist "test_records.csv" (
    echo test_records.csv is missing. Put it in the same folder as this file.
    goto :end
)

if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Run setup_and_train.bat first.
    goto :end
)

echo ==^> Retraining models from test_records.csv...
".venv\Scripts\python.exe" train_model.py
if errorlevel 1 (
    echo Retraining failed. Check the error above.
    goto :end
)

echo.
echo DONE. Updated complaint_pipeline.joblib and comparison charts.
echo Restart the Streamlit app to load the new model.
goto :end

:directory_error
echo Could not open the app folder.

:end
echo.
pause
