@echo off
echo Installing LousyBookML...

:: Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
pip install -r requirements.txt

:: Install package in development mode
pip install -e .

echo.
echo Installation complete! You can now use LousyBookML.
echo To run tests, use: python -m pytest tests/
echo.

pause
