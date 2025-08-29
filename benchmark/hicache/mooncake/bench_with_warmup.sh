SEED=42
N_PROMPTS=256
IN_LEN=2048
OUT_LEN=64
BACKEND=sglang
HOST=0.0.0.0
PORT=33301

COMMON="python3 -m sglang.bench_serving \
  --backend ${BACKEND} \
  --host ${HOST} --port ${PORT} \
  --dataset-name random \
  --num-prompts ${N_PROMPTS} \
  --random-input-len ${IN_LEN} \
  --random-output-len ${OUT_LEN} \
  --random-range-ratio 0.0 \
  --seed ${SEED} \
  --disable-tqdm \
  --model DeepSeek-R1-Distill-Qwen-7B \
  --dataset-path /home/yk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

${COMMON} --output-file ./benchmark_results/kv_run_pass1.jsonl

# ${COMMON} --output-file ./benchmark_results/kv_run_pass2.jsonl
