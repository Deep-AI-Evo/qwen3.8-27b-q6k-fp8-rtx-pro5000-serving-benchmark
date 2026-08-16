@echo off
chcp 65001 >nul
REM ============================================================
REM  限制 GPU 最大功率为 270W（需管理员权限，自动提权）
REM ============================================================
net session >nul 2>&1
if errorlevel 1 (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    if errorlevel 1 (
        echo  [Power] 错误: 提权失败，无法限制 GPU 功率，中止启动
        pause
        exit /b 1
    )
    exit /b
)
nvidia-smi -pl 270 >nul 2>&1
if %errorlevel% equ 0 (
    echo  [Power] OK: 已限制最大功率为 270W
) else (
    echo  [Power] 错误: 设置 270W 功率限制失败，为避免过热中止启动
    pause
    exit /b 1
)
echo.
set LLAMA_DIR=H:\llamacpp\llama-b9692-bin-win-cuda-13.3-x64
set MODEL=H:\models\Qwen3.8-27B-NVFP4-MTP-VERY-HIGH.gguf
set MMPROJ=H:\models\mmproj-Qwen3.8-27B-BF16.gguf

title Qwen3.8-27B NVFP4-MTP Vision Server
echo ================================================
echo  Qwen3.8-27B NVFP4-MTP Vision Server (FP4 GGUF)
echo  -----------------------------------------------
echo  Port      : 12345
echo  API       : http://localhost:12345/v1
echo  Model     : Qwen3.8-27B-NVFP4-MTP-VERY-HIGH.gguf
echo  Quant     : FP4 (Blackwell native) + MTP draft
echo  Speed     : decode ~68 t/s single (MTP on)
echo  Ctx Size  : 4 x 262144 (q8_0 KV)
echo  ================================================
echo.

"%LLAMA_DIR%\llama-server.exe" ^
  --model "%MODEL%" ^
  --mmproj "%MMPROJ%" ^
  --image-min-tokens 1024 ^
  --host 0.0.0.0 ^
  --port 12345 ^
  --ctx-size 1048576 ^
  --n-gpu-layers 99 ^
  --threads 8 ^
  --parallel 4 ^
  --flash-attn on ^
  --no-mmap ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --temp 1.0 ^
  --presence-penalty 0.0 ^
  --repeat-penalty 1.0 ^
  --top-p 0.95 ^
  --top-k 20 ^
  --min-p 0.00 ^
  --reasoning auto ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 4

pause
