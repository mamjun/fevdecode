@echo off
setlocal

set "REPO_ROOT=%~dp0"
set "PY_EXE=%REPO_ROOT%.venv\Scripts\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

set "FEV_PATH="
set "NAME="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--name" (
  set "NAME=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--fev" (
  set "FEV_PATH=%~2"
  shift
  shift
  goto parse_args
)
if not defined FEV_PATH (
  set "FEV_PATH=%~1"
)
shift
goto parse_args

:args_done
if defined NAME (
  set "ARG_LABEL=--name"
  set "ARG_VALUE=%NAME%"
) else (
  if defined FEV_PATH (
    set "ARG_LABEL=--fev"
    set "ARG_VALUE=%FEV_PATH%"
  ) else (
    set "ARG_LABEL=--name"
    set "ARG_VALUE=dontstarve_DLC003"
  )
)

echo.
echo About to run:
echo "%PY_EXE%" -m fevdecode gen-all %ARG_LABEL% "%ARG_VALUE%"
set /p CONFIRM=Continue? (y/N): 
if /I not "%CONFIRM%"=="y" if /I not "%CONFIRM%"=="yes" (
  echo Aborted.
  exit /b 1
)

"%PY_EXE%" -m fevdecode gen-all %ARG_LABEL% "%ARG_VALUE%"
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
