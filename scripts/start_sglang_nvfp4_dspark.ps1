docker rm -f sglang-nvfp4-dspark 2>$null
docker run -d --name sglang-nvfp4-dspark --gpus all -p 12345:12345 `
  -v "H:\models\RadixArk-Qwen3.8-27B-NVFP4:/models/nvfp4" `
  -v "H:\models\Qwen3.8-27B-DSpark:/models/dspark" `
  lmsysorg/sglang:qwen38-27b `
  python3 -m sglang.launch_server --model /models/nvfp4 `
  --speculative-algorithm DSPARK --speculative-draft-model-path /models/dspark `
  --speculative-dspark-block-size 7 --mamba-full-memory-ratio 0.7 `
  --chunked-prefill-size 2048 --reasoning-parser qwen3 --tool-call-parser qwen3_coder `
  --trust-remote-code --host 0.0.0.0 --port 12345
