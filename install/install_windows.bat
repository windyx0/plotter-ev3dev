@echo off
setlocal

echo.
echo [Plott3r] Installing base dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [Plott3r] Base install failed.
  exit /b 1
)

echo.
for /f "delims=" %%i in ('python -c "import sys,platform;print(f'cp{sys.version_info[0]}{sys.version_info[1]}-abi3-win_amd64') if platform.machine().lower().endswith('64') else print(f'cp{sys.version_info[0]}{sys.version_info[1]}-abi3-win32')"') do set PY_TAG=%%i
echo [Plott3r] Your expected OpenCV wheel tag: %PY_TAG%
echo.
echo [Plott3r] You can install CUDA OpenCV from:
echo  - Local *.whl file
echo  - Direct URL (for example from:
echo    https://github.com/cudawarped/opencv-python-cuda-wheels/releases)
echo.
choice /C YN /M "Install CUDA OpenCV build now?"
if errorlevel 2 goto done
if errorlevel 1 goto cuda

:cuda
echo.
echo [Plott3r] CUDA install selected.
echo Paste local wheel path OR direct wheel URL, then press Enter.
echo Example path: C:\wheels\opencv_python-4.x-cp310-cp310-win_amd64.whl
echo Example URL : https://github.com/.../opencv_python-....whl
set /p CUDA_WHL=Wheel path or URL:

if "%CUDA_WHL%"=="" (
  echo.
  echo [Plott3r] No input provided. Keeping CPU OpenCV.
  goto done
)

set IS_URL=0
echo %CUDA_WHL% | findstr /I /B "http:// https://"
if not errorlevel 1 set IS_URL=1

echo.
echo [Plott3r] Replacing CPU OpenCV with CUDA wheel...
python -m pip uninstall -y opencv-python opencv-contrib-python >nul 2>nul

if "%IS_URL%"=="1" (
  python -m pip install "%CUDA_WHL%"
) else (
  if not exist "%CUDA_WHL%" (
    echo.
    echo [Plott3r] Wheel file not found: %CUDA_WHL%
    echo [Plott3r] Keeping CPU OpenCV.
    goto done
  )
  python -m pip install "%CUDA_WHL%"
)

if errorlevel 1 (
  echo.
  echo [Plott3r] CUDA wheel install failed. Restoring CPU OpenCV...
  python -m pip install "opencv-python>=4.8,<5"
  goto done
)

echo.
echo [Plott3r] CUDA wheel installed successfully.
echo [Plott3r] Checking CUDA availability in cv2...
python -c "import cv2; ok=hasattr(cv2,'cuda') and cv2.cuda.getCudaEnabledDeviceCount()>0; print('[Plott3r] cv2 CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2,'cuda') else 0); raise SystemExit(0 if ok else 1)"
if errorlevel 1 (
  echo [Plott3r] WARNING: OpenCV installed, but CUDA is NOT available at runtime.
  echo [Plott3r] Check your wheel/Python/CUDA/driver compatibility.
  echo [Plott3r] Rolling back to CPU OpenCV to keep app working...
  python -m pip uninstall -y opencv-python opencv-contrib-python >nul 2>nul
  python -m pip install "opencv-python>=4.8,<5"
) else (
  echo [Plott3r] CUDA runtime check passed.
)

:done
echo.
echo [Plott3r] Done.
endlocal
