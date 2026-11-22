@echo off
REM Launch the Intelligent Document Classification System (Windows 11)
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
if not exist models\document_classifier.keras (
    echo.
    echo [WARNING] models\document_classifier.keras not found.
    echo The app will open but cannot classify until you train the model:
    echo     python -m src.data_loader
    echo     python -m src.train
    echo     python -m src.evaluate
    echo.
)
streamlit run app\Home.py
pause
