@echo off
setlocal
cd /d "%~dp0\.."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
pyinstaller packaging\CellWellSegmentation.spec --clean --noconfirm
pause
