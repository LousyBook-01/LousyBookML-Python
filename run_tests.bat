@echo off
echo Running LousyBookML tests...

:: Run tests with coverage report
python -m pytest tests/ -v --cov=LousyBookML --cov-report=term-missing

echo.
echo Test run complete!
echo.

pause
