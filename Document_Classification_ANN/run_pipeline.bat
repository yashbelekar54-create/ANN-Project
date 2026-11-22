@echo off
REM Full pipeline: data -> train -> metadata comparison -> evaluate -> tests (Windows 11)
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python -m src.data_loader || goto :error
python -m src.train --compare-metadata || goto :error
python -m src.evaluate || goto :error
python -m pytest tests -v || goto :error
echo.
echo Pipeline finished. Launch the app with run_app.bat
pause
exit /b 0
:error
echo.
echo Pipeline stopped because a step failed.
pause
exit /b 1
