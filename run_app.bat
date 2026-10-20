@echo off
rem Starts the Customer Segmentation and CLV web app; it opens in your browser at http://localhost:8501
rem Close this window to stop the app.
setlocal
title Customer Segmentation and CLV app
cd /d "%~dp0"
rem do not write Python cache files into this (OneDrive) folder
set PYTHONDONTWRITEBYTECODE=1

rem ---- 1) uv: fast, and it keeps the libraries in its own cache (outside OneDrive) ----
set "UV="
for /f "delims=" %%i in ('where uv 2^>nul') do if not defined UV set "UV=%%i"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV goto :pip
echo Starting the app with uv. The first start downloads the libraries: about 1-2 minutes.
"%UV%" run --no-project --python 3.13 --with-requirements requirements.txt -- streamlit run app.py
goto :end

rem ---- 2) no uv: a private Python environment in %LOCALAPPDATA% (created once) ----
:pip
set "VENV=%LOCALAPPDATA%\clv_segmentation_app_venv"
if exist "%VENV%\Scripts\python.exe" goto :run
echo Creating a Python environment in %VENV% - first time only, a few minutes...
py -3.13 -m venv "%VENV%" 2>nul
if not exist "%VENV%\Scripts\python.exe" python -m venv "%VENV%" 2>nul
if not exist "%VENV%\Scripts\python.exe" goto :no_python
"%VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
:run
"%VENV%\Scripts\python.exe" -m streamlit run app.py
goto :end

:no_python
echo Python 3.13 was not found. Install it from https://www.python.org/downloads/ and run this file again.

:end
pause
