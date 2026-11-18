@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app.py %*
echo.
echo App da thoat. Nhan phim bat ky de dong cua so...
pause >nul

