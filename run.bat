@echo off
setlocal
cd /d "%~dp0"
set PORT=8000

where py >nul 2>nul
if not errorlevel 1 (
  set PY=py -3
  goto :serve
)
where python >nul 2>nul
if not errorlevel 1 (
  set PY=python
  goto :serve
)

echo Python 3 was not found.
echo You can still open index.html directly by double-clicking it,
echo or install Python 3 from https://www.python.org/downloads/ and run this file again.
pause
goto :end

:serve
echo Starting How AI Learns to See on http://localhost:%PORT%
echo Close this window to stop the server.
start "" "http://localhost:%PORT%/"
%PY% -m http.server %PORT% --bind 127.0.0.1

:end
endlocal
