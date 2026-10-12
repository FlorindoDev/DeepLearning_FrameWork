@echo off
setlocal

set "AF_VERSION=3.10.0"
set "AF_URL=https://arrayfire.gateway.scarf.sh/windows/%AF_VERSION%/ArrayFire.exe"
for %%I in ("%~dp0..\deps\downloads") do set "AF_DOWNLOAD_DIR=%%~fI"
set "AF_INSTALLER=%AF_DOWNLOAD_DIR%\ArrayFire-v%AF_VERSION%-Windows-x64.exe"

if /I "%~1"=="--help" (
    echo Scarica l'installer ufficiale ArrayFire Windows x64 in deps\downloads.
    echo L'installazione va avviata manualmente dopo il download.
    exit /b 0
)

if not exist "%AF_DOWNLOAD_DIR%" (
    mkdir "%AF_DOWNLOAD_DIR%"
    if errorlevel 1 exit /b 1
)

if exist "%AF_INSTALLER%" goto pronto
where curl.exe >nul 2>&1
if errorlevel 1 (
    echo Errore: curl.exe non e disponibile nel PATH.
    exit /b 1
)

echo Download ArrayFire %AF_VERSION% per Windows x64...
curl.exe --fail --location --retry 3 --output "%AF_INSTALLER%.part" "%AF_URL%"
if errorlevel 1 (
    echo Download fallito. Riprova con make setup-windows.
    exit /b 1
)
move /Y "%AF_INSTALLER%.part" "%AF_INSTALLER%" >nul
if errorlevel 1 exit /b 1

:pronto
echo Installer disponibile: "%AF_INSTALLER%"
echo Per installare ArrayFire, esegui:
echo "%AF_INSTALLER%"
echo Dopo l'installazione, riapri il terminale per aggiornare AF_PATH e PATH.
exit /b 0
