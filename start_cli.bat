@echo off
REM ============================================================
REM  synthfp - command line helper (Windows)
REM ============================================================
REM  Runs the CLI with the first Python that has the core
REM  packages. Without arguments it opens an interactive console.
REM
REM  Examples:
REM      start_cli.bat selfcheck
REM      start_cli.bat generate --preset quick_test
REM      start_cli.bat stats --dataset outputs\synth_fp_v1
REM      start_cli.bat                     (interactive)
REM ============================================================
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

set "PYEXE="
set "PYRUN="

call :probe "%~dp0.venv\Scripts\python.exe"
call :probe "%~dp0venv\Scripts\python.exe"
call :probe "%~dp0env\Scripts\python.exe"
call :probe "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
call :probe "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :probe "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
call :probe "%ProgramFiles%\Python313\python.exe"
call :probe "%ProgramFiles%\Python312\python.exe"
call :probe "C:\Python313\python.exe"
call :probe "C:\Python312\python.exe"
for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYEXE call :probe "%%~P"
)
if not defined PYEXE (
    py -3 -c "import numpy, cv2, pydantic, yaml" >nul 2>nul
    if not errorlevel 1 (
        set "PYEXE=py -3"
        set "PYRUN=py -3"
    )
)

if not defined PYEXE (
    echo.
    echo  [synthfp] Could not find a Python with the required packages.
    echo   1^) Install Python 3.10+ ^(tick "Add python.exe to PATH"^)
    echo   2^) python -m pip install -r requirements.txt
    echo.
    popd
    exit /b 1
)

echo [synthfp] python : %PYEXE%
echo.

if "%~1"=="" (
    echo [synthfp] interactive console -- try:  python -m synthfp --help
    echo.
    cmd /k "%PYEXE% -m synthfp --help"
    popd
    exit /b 0
)

%PYRUN% -m synthfp %*
set "RC=%ERRORLEVEL%"
popd
endlocal & exit /b %RC%

:probe
if defined PYEXE goto :eof
if not exist "%~1" goto :eof
"%~1" -c "import numpy, cv2, pydantic, yaml" >nul 2>nul
if errorlevel 1 goto :eof
set "PYEXE=%~1"
set "PYRUN="%~1""
goto :eof
