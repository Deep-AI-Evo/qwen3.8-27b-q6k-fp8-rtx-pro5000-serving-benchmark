docker rm -f sglang-fp8-dspark 2>$null
docker run -d --name sglang-fp8-dspark --gpus all -p 12348:12345 `
  -v "H:\models\Qwen3.8-27B-FP8:/models/fp8" `
  -v "H:\models\Qwen3.8-27B-DSpark:/models/dspark" `
  lmsysorg/sglang:qwen38-27b `
  python3 -m sglang.launch_server --model /models/fp8 `
  --speculative-algorithm DSPARK --speculative-draft-model-path /models/dspark `
  --speculative-dspark-block-size 7 --mamba-full-memory-ratio 0.7 `
  --chunked-prefill-size 2048 --reasoning-parser qwen3 --tool-call-parser qwen3_coder `
  --mm-feature-transport cpu `
  --trust-remote-code --host 0.0.0.0 --port 12345
