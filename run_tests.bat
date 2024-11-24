@echo off
echo Running LousyBookML tests...

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Run tests with coverage report
python -m pytest tests/ -v --cov=LousyBookML --cov-report=term-missing

echo.
echo Test run complete!
echo.

pause
