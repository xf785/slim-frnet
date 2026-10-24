@echo off
REM ============================================================
REM  synthfp - one-click GUI launcher (Windows)
REM ============================================================
REM  Picks the first Python on this machine that can actually
REM  import the core packages, then starts the graphical UI.
REM
REM  Invocation goes through %PYRUN% which always carries the quotes
REM  a path with spaces needs (PYEXE stays unquoted for display).
REM
REM  ASCII only on purpose: cmd.exe reads .bat files with the OEM
REM  code page, so non-ASCII comments can corrupt the script.
REM ============================================================
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

set "PYEXE="
set "PYRUN="

REM --- candidates, in preference order -------------------------------
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

REM --- then whatever `where python` finds ---------------------------
for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYEXE call :probe "%%~P"
)
REM --- finally the py launcher --------------------------------------
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
    echo.
    echo   1^) Install Python 3.10+ from https://www.python.org/downloads/
    echo      ^(tick "Add python.exe to PATH"^)
    echo   2^) Install the dependencies:
    echo        python -m pip install -r requirements.txt
    echo.
    pause
    popd
    exit /b 1
)

echo [synthfp] python : %PYEXE%

REM --- a Qt binding must exist (PySide6 preferred, PyQt6 accepted) ---
%PYRUN% -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    %PYRUN% -c "import PyQt6" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo  [synthfp] No Qt binding found. Install one of them:
        echo        "%PYEXE%" -m pip install PySide6
        echo        "%PYEXE%" -m pip install PyQt6
        echo.
        pause
        popd
        exit /b 1
    )
)

echo [synthfp] starting GUI ...
%PYRUN% -m synthfp gui %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo  [synthfp] GUI exited with code %RC%.
    echo            Diagnose with:  "%PYEXE%" -m synthfp selfcheck
    echo.
    pause
)

popd
endlocal & exit /b %RC%

REM ------------------------------------------------------------------
REM  :probe  <python-path>
REM  Sets PYEXE/PYRUN when the candidate exists and imports core deps.
REM ------------------------------------------------------------------
:probe
if defined PYEXE goto :eof
if not exist "%~1" goto :eof
"%~1" -c "import numpy, cv2, pydantic, yaml" >nul 2>nul
if errorlevel 1 goto :eof
set "PYEXE=%~1"
set "PYRUN="%~1""
goto :eof
