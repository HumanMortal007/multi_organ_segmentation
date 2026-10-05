@echo off
echo ============================================================
echo  2D U-Net Multi-Organ Segmentation - Environment Setup
echo ============================================================

echo.
echo [1] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

echo.
echo [2] Upgrading pip...
venv\Scripts\pip.exe install --upgrade pip

echo.
echo [3] Installing PyTorch with CUDA 12.1 support...
venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo [4] Installing other dependencies...
venv\Scripts\pip.exe install numpy scipy h5py matplotlib pyyaml tqdm medpy SimpleITK

echo.
echo [5] Verifying CUDA availability...
venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB' if torch.cuda.is_available() else '')"

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Run training:
echo    venv\Scripts\python.exe train.py --epochs 5
echo.
echo  Main experiment:
echo    venv\Scripts\python.exe train.py --epochs 20
echo.
echo  Extended run:
echo    venv\Scripts\python.exe train.py --epochs 50
echo.
echo  Resume training:
echo    venv\Scripts\python.exe train.py --resume checkpoints/last_checkpoint.pth --epochs 20
echo.
echo  Evaluate only:
echo    venv\Scripts\python.exe evaluate.py --checkpoint checkpoints/best_model.pth --save_vis
echo ============================================================
pause
