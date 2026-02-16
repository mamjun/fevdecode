@echo off
setlocal

set "REPO_ROOT=%~dp0"
set "PY_EXE=%REPO_ROOT%.venv\Scripts\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

set "FEV_PATH=%~1"
if "%FEV_PATH%"=="" set "FEV_PATH=%REPO_ROOT%sound\dontstarve_DLC003.fev"

"%PY_EXE%" -m fevdecode gen-all --fev "%FEV_PATH%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo gen-all failed with exit code %EXIT_CODE%.
  pause
  exit /b %EXIT_CODE%
)

echo.
echo gen-all completed successfully.
pause
endlocal
