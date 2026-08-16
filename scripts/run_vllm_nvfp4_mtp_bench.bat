@echo off
chcp 65001 >nul
set VENV=H:\vllm-win
set MODEL_DIR=H:\models\Qwen3.8-27B-NVFP4
set PORT=12345
set VLLM_TEST_FORCE_FP8_MARLIN=1
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKV=10.0.26100.0
set PATH=%MSVC%\bin\HostX64\x64;%SDK%\bin\%SDKV%\x64;%VENV%\Scripts;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKV%\ucrt;%SDK%\Include\%SDKV%\um;%SDK%\Include\%SDKV%\shared;%SDK%\Include\%SDKV%\winrt
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKV%\ucrt\x64;%SDK%\Lib\%SDKV%\um\x64
set CUDA_HOME=%VENV%\Lib\site-packages\nvidia\cu13
set CUDA_PATH=%CUDA_HOME%
set CUDA_LIB_PATH=%CUDA_HOME%\lib
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\bin\x86_64;%PATH%

REM ============================================================
REM  Qwen3.8 official recommended params (Thinking Mode):
REM    temperature=1.0, top_p=0.95, top_k=20, min_p=0.0
REM    presence_penalty=0.0, repetition_penalty=1.0
REM  Thinking mode is ON by default (enable_thinking, preserve_thinking)
REM  vLLM sampling params are request-level; clients should pass:
REM    {"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,
REM     "presence_penalty":0.0,"repetition_penalty":1.0}
REM ============================================================
"%VENV%\Scripts\python.exe" -m vllm.entrypoints.openai.api_server ^
  --model "%MODEL_DIR%" ^
  --served-model-name Qwen3.8-27B-NVFP4-MTP ^
  --linear-backend marlin ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --max-model-len 262144 ^
  --gpu-memory-utilization 0.90 ^
  --max-num-seqs 4 ^
  --enable-prefix-caching ^
  --tool-call-parser qwen3_xml ^
  --enable-auto-tool-choice ^
  --default-chat-template-kwargs "{\"enable_thinking\": false}" ^
  --speculative-config "{\"num_speculative_tokens\": 1, \"method\": \"mtp\"}" ^
  --trust-remote-code

pause
