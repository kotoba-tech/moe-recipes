#!/bin/bash

set -e

# swich virtual env
source .env/bin/activate



start=500
end=61000
increment=500

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/model/MoE/checkpoint/Mixtral-8x1.3B/lr_1e-4-minlr_1e-6_warmup_2000_seq_4096/${FORMATTED_ITERATION}/fp32_model.bin
  OUTPUT_PATH=/model/MoE/converted-hf-checkpoint/Mixtral-8x1.3B/lr_1e-4-minlr_1e-6_warmup_2000_seq_4096/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/home/taishi/moe-recipes/tools/checkpoint-convert/scripts/mdx/Mixtral-8x1.3B-llm-jp/config.json

  python tools/checkpoint-convert/scripts/mdx/convert_ckpt.py \
    --config $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length 4096
done
