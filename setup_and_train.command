#!/bin/bash
# STEP 1 (Mac): installs everything and trains the model. Run this once.
# Double-click it, or in Terminal:  bash setup_and_train.command

cd "$(dirname "$0")" || exit 1

pause() { echo; read -n 1 -s -r -p "Press any key to close this window..."; echo; }

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found."
  echo "Install it from https://www.python.org/downloads/macos/ then run this again."
  pause; exit 1
fi

if [ ! -f test_records.csv ]; then
  echo "test_records.csv is missing. Put it in the same folder as this file."
  pause; exit 1
fi

echo "==> Creating a private Python environment (.venv)..."
python3 -m venv .venv || { echo "Could not create the environment."; pause; exit 1; }
source .venv/bin/activate

echo "==> Installing packages (this can take a few minutes the first time)..."
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt || { echo "Install failed. Check your internet connection."; pause; exit 1; }

echo "==> Training the models..."
python train_model.py || { echo "Training failed. Copy the error above and ask for help."; pause; exit 1; }

echo
echo "DONE. Created in this folder:"
echo "  complaint_pipeline.joblib, model_comparison.csv/.png, confusion_*.png"
echo "Next: double-click run_app.command"
pause
