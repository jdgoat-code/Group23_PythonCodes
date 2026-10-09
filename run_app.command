#!/bin/bash
# STEP 2 (Mac): opens the Streamlit interface in your browser.
# Double-click it, or in Terminal:  bash run_app.command
# Stop the app any time with Control + C.

cd "$(dirname "$0")" || exit 1

pause() { echo; read -n 1 -s -r -p "Press any key to close this window..."; echo; }

if [ ! -d .venv ]; then
  echo "Setup has not been run yet. Double-click setup_and_train.command first."
  pause; exit 1
fi

if [ ! -f complaint_pipeline.joblib ]; then
  echo "No trained model found. Double-click setup_and_train.command first."
  pause; exit 1
fi

source .venv/bin/activate
echo "Starting the app... your browser will open at http://localhost:8501"
python -m streamlit run app.py
pause
