@echo off
cd /d "%~dp0"
python -c "import cv2, numpy" 2>nul
if errorlevel 1 (
  echo First run: installing requirements, this takes a minute...
  python -m pip install -r requirements.txt
  if errorlevel 1 goto fail
)
python -m localface
if errorlevel 1 goto fail
exit /b 0
:fail
echo.
echo Something went wrong. Check that Python 3.9+ is installed ^(with "Add Python to PATH" ticked^) and that you are online.
pause
