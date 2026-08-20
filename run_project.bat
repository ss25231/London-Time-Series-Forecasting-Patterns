@echo off
REM ============================================================================
REM  run_project.bat  —  one-click: install dependencies, then run everything.
REM
REM  Usage (from the project root, e.g. C:\Users\ashre\Documents\project):
REM      run_project.bat
REM      run_project.bat --quick        (fast pass, small bootstrap)
REM
REM  It will:
REM    1. install the required Python packages
REM    2. run the whole pipeline via run_all.py (all stages, in order)
REM    3. print a timed summary at the end
REM ============================================================================

setlocal

REM --- pick a Python launcher: prefer the 'py' launcher, else 'python' -------
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

echo.
echo ============================================================
echo   STEP 1 of 2 — installing Python dependencies
echo ============================================================
%PY% -m pip install --upgrade pip
%PY% -m pip install pandas numpy scikit-learn lightgbm matplotlib pyarrow python-docx openpyxl python-pptx
if %errorlevel% neq 0 (
    echo.
    echo !! Dependency installation failed. Fix the error above and re-run.
    exit /b 1
)

echo.
echo ============================================================
echo   STEP 2 of 2 — running the full pipeline
echo ============================================================

REM run_all.py finds scripts in .\scripts or alongside itself.
if exist "scripts\run_all.py" (
    %PY% scripts\run_all.py %*
) else (
    %PY% run_all.py %*
)

echo.
echo ============================================================
echo   Done. See the summary above and pipeline_runtime.txt.
echo ============================================================

endlocal
