@echo off
chcp 65001 >nul
set LLAMA_DIR=H:\llamacpp\llama-b9692-bin-win-cuda-13.3-x64
set MODEL=H:\models\Qwen3.8-27B-UD-Q6_K_XL.gguf
set MMPROJ=H:\models\mmproj-Qwen3.8-27B-BF16.gguf

title Qwen3.8-27B Vision Server
echo ================================================
echo  Qwen3.8-27B Vision Server (Multimodal)
echo  -----------------------------------------------
echo  Port      : 12345
echo  API       : http://localhost:12345/v1
echo  Web UI    : http://localhost:12345
echo  Model     : Qwen3.8-27B-UD-Q6_K_XL.gguf
echo  Vision    : mmproj-Qwen3.8-27B-BF16.gguf
echo  GPU       : RTX PRO 5000 72GB (all layers offloaded)
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
  --reasoning auto

pause
